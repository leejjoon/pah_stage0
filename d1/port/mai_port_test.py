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
import sys, time, glob, os
import numpy as np

root, xyzdir = sys.argv[1], sys.argv[2]
sweep_s = float(sys.argv[3]) if len(sys.argv) > 3 else 600
shard, n_shards = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mai_port import Port, MAX_NEIGHBORS
m = Port(root)
import jax, jax.numpy as jnp
from neuralil.bessel_descriptors import get_max_number_of_neighbors
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

print("jax", jax.__version__, jax.devices())
ff = m.ff
for name, mi in (("NNFF", m.ff), ("EPNN", m.dp)):
    leaves = jax.tree.leaves(mi.params)
    print(f"[1] {name}: {sum(np.size(x) for x in leaves)} parameters, dtypes {sorted({str(np.asarray(x).dtype) for x in leaves})}")
    print("    ", {k: v for k, v in vars(mi).items() if k != "params" and not hasattr(v, "shape")})

def dipole(atoms):
    mu, q = m.dipole(atoms)
    return mu, q.sum()

def eval_on(device, atoms):
    with jax.default_device(device):
        calc = m.calculator(n_devices=1)
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
    a = naph.copy(); a.calc = m.calculator(n_devices=1)
    MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.default_rng(0))
    md = VelocityVerlet(a, 0.5 * units.fs); etot = []
    md.attach(lambda: etot.append(a.get_potential_energy() + a.get_kinetic_energy()), interval=20)
    t0 = time.perf_counter(); md.run(2000); dt = time.perf_counter() - t0
    etot = np.array(etot)
    print(f"[3] NVE 1 ps naphthalene, 300 K start: E_tot drift {etot[-1] - etot[0]:+.2e} eV, "
          f"std {etot.std():.2e} eV; {dt / 2000 * 1e3:.2f} ms per step (ASE, one molecule)")

files = sorted(glob.glob(f"{xyzdir}/*.xyz"))[shard::n_shards]; bad, maxf, rows, n_done, t_start = [], [], [], 0, time.perf_counter()
with jax.default_device(gpu):
    calc = m.calculator(n_devices=1)
    for fn in files:
        if time.perf_counter() - t_start > sweep_s: break
        a = read(fn); name = os.path.basename(fn)[:-4]
        if set(a.get_chemical_symbols()) - set(ff.sorted_elements):
            bad.append((name, "element")); n_done += 1; continue
        nn = int(get_max_number_of_neighbors(jnp.asarray(a.positions), jnp.asarray([calc.symbol_map[s] for s in a.symbols]),
                                             ff.r_cut, jnp.asarray(a.cell[...])))
        if nn > MAX_NEIGHBORS: bad.append((name, f"neighbours {nn} > {MAX_NEIGHBORS}"))
        a.calc = calc
        try:
            e, f = a.get_potential_energy(), a.get_forces()
            if not (np.isfinite(e) and np.isfinite(f).all()): bad.append((name, "nonfinite"))
            fn_ = np.linalg.norm(f, axis=1); i = int(fn_.argmax())
            maxf.append(fn_[i]); rows.append((fn_[i], name, a.get_chemical_formula(), len(a), a[i].symbol, i))
        except Exception as ex:
            bad.append((name, f"{type(ex).__name__}: {str(ex)[:80]} (neighbours {nn})"))
        n_done += 1
maxf = np.array(maxf)
print(f"[4] shard {shard}/{n_shards} sweep: {n_done}/{len(files)} geometries in {time.perf_counter() - t_start:.0f} s; {len(bad)} problems; "
      f"max|F| (per-atom norm) at DFT geometry: median {np.median(maxf):.3f}, 95% {np.percentile(maxf, 95):.3f}, max {maxf.max():.3f} eV/A")
for b in bad[:20]: print("    ", b)
print(f"[4] largest max|F| (eV/A, per-atom norm): species, formula, atoms, element and index of that atom")
for r in sorted(rows, reverse=True)[:15]: print(f"     {r[0]:7.3f}  {r[1]}  {r[2]}  {r[3]}  {r[4]}{r[5]}")
