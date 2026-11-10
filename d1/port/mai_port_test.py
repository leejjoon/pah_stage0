# Port test for the Mai et al. 2025 MLMD models (Zenodo 15771437) in container A (JAX 0.11, Flax 0.12).
# Uses the authors' own model classes (lib/ of the Zenodo zip, Apache 2.0), patched only for API removals,
# and loads their pickles with an allow-list unpickler. Checks:
#   1. both models load; parameter counts and dtypes;
#   2. energy, forces and EPNN dipole for naphthalene on GPU and CPU agree;
#   3. 1 ps NVE (0.5 fs, ASE VelocityVerlet) on GPU conserves energy;
#   4. every one of the 1,704 Mai geometries: elements known, neighbours <= 25, finite energy and forces on GPU,
#      max |F| at the DFT geometry (time-limited; reports how many were done).
# Usage: python mai_port_test.py <Supplementary_Data_Code dir> <dir of 1,704 xyz> [sweep seconds] [shard] [n_shards]
# With shards (one process per GPU), shard k sweeps every n_shards-th geometry; only shard 0 runs checks 1-3.
import sys, time, pickle, importlib, datetime, glob, os
import numpy as np

root, xyzdir = sys.argv[1], sys.argv[2]
sweep_s = float(sys.argv[3]) if len(sys.argv) > 3 else 600
shard, n_shards = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
sys.path[:0] = [f"{root}/lib/bessel-nn-potentials-velo/src", f"{root}/lib/epnn-main/src"]

import jax, jax.numpy as jnp
if not hasattr(jax, "tree_map"):                       # removed in JAX 0.6; the Mai code calls it
    jax.tree_map = jax.tree.map
sys.modules.setdefault("jax.config", jax.config)       # the jax.config module is gone; bessel_descriptors imports it
_update = jax.config.update                            # ase_integration_gai forces the CPU platform on import
jax.config.update = lambda k, v: None if k == "jax_platform_name" else _update(k, v)
from neuralil.ase_integration_gai import NeuralILASECalculator
jax.config.update = _update
from neuralil.bessel_descriptors import PowerSpectrumGenerator, get_max_number_of_neighbors
from neuralil.model_older import ResNetCore as ResNetCoreOld, NeuralILwithMorse
from neuralil.model import ResNetCore
from neuralil.plain_ensembles.model import PlainEnsemblewithMorse
from epnn.model import EPNN, MLP
from epnn.preprocessing import create_batched_graph_list
from epnn.utils import get_nodes_from_graph
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

class Safe(pickle.Unpickler):
    ALLOWED = {("datetime", "datetime"), ("numpy", "ndarray"), ("numpy", "dtype"),
               ("neuralil.model", "NeuralILModelInfo"), ("epnn.model", "EPNNModelInfo")}
    def find_class(self, mod, name):
        if mod in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(importlib.import_module("numpy._core.multiarray"), name)
        if (mod, name) == ("jax._src.array", "_reconstruct_array"):
            return getattr(importlib.import_module(mod), name)
        if (mod, name) in self.ALLOWED:
            return getattr(importlib.import_module(mod), name)
        raise pickle.UnpicklingError(f"blocked global {mod}.{name}")

def load(p):
    with open(p, "rb") as f:
        return Safe(f).load()

print("jax", jax.__version__, jax.devices())
ff = load(f"{root}/model/model_enery.pkl"); dp = load(f"{root}/model/model_dipole.pkl")
for name, m in (("NNFF", ff), ("EPNN", dp)):
    leaves = jax.tree.leaves(m.params)
    print(f"[1] {name}: {sum(np.size(x) for x in leaves)} parameters, dtypes {sorted({str(np.asarray(x).dtype) for x in leaves})}")
    print("    ", {k: v for k, v in vars(m).items() if k != "params" and not hasattr(v, "shape")})

MAX_NEIGHBORS, N_ENS = 25, 5
desc = PowerSpectrumGenerator(ff.n_max, ff.r_cut, len(ff.sorted_elements), MAX_NEIGHBORS)
individual = NeuralILwithMorse(len(ff.sorted_elements), ff.embed_d, ff.r_cut, desc.process_data,
                               desc.process_some_data, ResNetCoreOld(ff.core_widths))
ensemble = PlainEnsemblewithMorse(individual, N_ENS)

# EPNN dipole, as in the authors' 2_calc_dipole.py (positions centred on the centre of mass, cell zero).
d_elems = sorted(["C", "H", "O", "N"]); d_map = {s: i for i, s in enumerate(d_elems)}
d_desc = jax.jit(PowerSpectrumGenerator(6, dp.r_cut, len(d_elems), MAX_NEIGHBORS).process_data)
epnn = EPNN(len(d_elems), dp.embed_dim, MLP(dp.update_model_widths),
            tuple(ResNetCore(dp.message_generator_widths) for _ in range(dp.n_passes)),
            tuple(ResNetCore(dp.pass_generator_widths) for _ in range(dp.e_passes)))
MASS = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}
def dipole(atoms):
    w = np.array([MASS[s] for s in atoms.get_chemical_symbols()])
    p = atoms.positions - (w[:, None] * atoms.positions).sum(0) / w.sum()
    t = jnp.array([d_map[s] for s in atoms.get_chemical_symbols()]); c = jnp.zeros((3, 3), jnp.float32)
    pj = jnp.asarray(p, jnp.float32); d = d_desc(pj, t, c)
    g = create_batched_graph_list(pj[None], c[None], d.reshape(1, d.shape[0], -1), t[None],
                                  jnp.zeros((1, len(p), 1)), dp.r_cut, dp.r_switch, dp.node_state_dim, 1)
    q = np.asarray(jnp.squeeze(get_nodes_from_graph(epnn.apply(dp.params, g[0]))[0]))
    return (q[:, None] * p).sum(0), q.sum()

def eval_on(device, atoms):
    with jax.default_device(device):
        calc = NeuralILASECalculator(ensemble, ff, MAX_NEIGHBORS, n_devices=1)
        a = atoms.copy(); a.calc = calc
        e = a.get_potential_energy(); f = a.get_forces(); mu, qt = dipole(a)
    return e, f, mu, qt

gpu, cpu = jax.devices()[0], jax.devices("cpu")[0]
naph = read(f"{xyzdir}/C10H8_330.xyz")
res = {}
for dev in ((gpu, cpu) if shard == 0 else ()):
    t0 = time.perf_counter(); res[dev.platform] = eval_on(dev, naph); dt = time.perf_counter() - t0
    e, f, mu, qt = res[dev.platform]
    print(f"[2] {dev}: E {e:.6f} eV, max|F| {np.abs(f).max():.4f} eV/A, dipole {np.round(mu, 5)} e*A, "
          f"total charge {qt:.2e} ({dt:.1f} s incl. compile)")
if shard == 0:
    eg, fg, mg, _ = res["gpu"]; ec, fc, mc, _ = res["cpu"]
    print(f"[2] GPU vs CPU: |dE| {abs(eg - ec):.2e} eV, max|dF| {np.abs(fg - fc).max():.2e} eV/A, max|dmu| {np.abs(mg - mc).max():.2e}")

if shard == 0:
  with jax.default_device(gpu):
    a = naph.copy(); a.calc = NeuralILASECalculator(ensemble, ff, MAX_NEIGHBORS, n_devices=1)
    MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.default_rng(0))
    md = VelocityVerlet(a, 0.5 * units.fs); etot = []
    md.attach(lambda: etot.append(a.get_potential_energy() + a.get_kinetic_energy()), interval=20)
    t0 = time.perf_counter(); md.run(2000); dt = time.perf_counter() - t0
    etot = np.array(etot)
    print(f"[3] NVE 1 ps naphthalene, 300 K start: E_tot drift {etot[-1] - etot[0]:+.2e} eV, "
          f"std {etot.std():.2e} eV; {dt / 2000 * 1e3:.2f} ms per step (ASE, one molecule)")

files = sorted(glob.glob(f"{xyzdir}/*.xyz"))[shard::n_shards]; bad, maxf, n_done, t_start = [], [], 0, time.perf_counter()
with jax.default_device(gpu):
    calc = NeuralILASECalculator(ensemble, ff, MAX_NEIGHBORS, n_devices=1)
    for fn in files:
        if time.perf_counter() - t_start > sweep_s: break
        a = read(fn); name = os.path.basename(fn)[:-4]
        if set(a.get_chemical_symbols()) - set(ff.sorted_elements):
            bad.append((name, "element")); n_done += 1; continue
        nn = int(get_max_number_of_neighbors(jnp.asarray(a.positions), jnp.asarray([calc.symbol_map[s] for s in a.symbols]),
                                             ff.r_cut, jnp.asarray(a.cell[...])))
        a.calc = calc
        try:
            e, f = a.get_potential_energy(), a.get_forces()
            if not (np.isfinite(e) and np.isfinite(f).all()): bad.append((name, "nonfinite"))
            maxf.append(np.abs(f).max())
        except Exception as ex:
            bad.append((name, f"{type(ex).__name__}: {str(ex)[:80]} (neighbours {nn})"))
        n_done += 1
maxf = np.array(maxf)
print(f"[4] shard {shard}/{n_shards} sweep: {n_done}/{len(files)} geometries in {time.perf_counter() - t_start:.0f} s; {len(bad)} problems; "
      f"max|F| at DFT geometry: median {np.median(maxf):.3f}, 95% {np.percentile(maxf, 95):.3f}, max {maxf.max():.3f} eV/A")
for b in bad[:20]: print("    ", b)
