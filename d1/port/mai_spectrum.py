# Step 2b: one species end to end with the port, following the authors' protocol (code/1_MD_calc_position.py,
# 2_calc_dipole.py), as a jitted loop on the GPU instead of ASE (about 10 h per molecule through ASE).
#   1. validate the jitted velocity Verlet against ASE VelocityVerlet (same positions and velocities, 200 steps);
#   2. R replicas (seeds) together: Maxwell-Boltzmann velocities, NVT Langevin (BAOAB, friction 0.02 per ASE time
#      unit = 1.96 ps^-1, centre-of-mass velocity removed every step as ASE's fixcm), then NVE velocity Verlet;
#      0.5 fs; positions every 2 steps, rounded to 1e-5 A as in the authors' text trajectories;
#   3. EPNN dipoles on centre-of-mass-centred frames (batched; checked against mai_port.Port.dipole);
#   4. writes <outdir>/run_s<k>/intermediate_results/Dipole_txt_mass/... in the authors' format, so their
#      3_calc_IR.py runs on it unmodified (in their environment), and <outdir>/md_s<k>.npz (energies, temperatures).
# Usage: python mai_spectrum.py <Supplementary_Data_Code dir> <xyz file> <outdir> [T=300] [R=3] [nvt=400000] [nve=400000]
import sys, os, time
import numpy as np

root, xyz, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
T = float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
R = int(sys.argv[5]) if len(sys.argv) > 5 else 3
N_NVT = int(sys.argv[6]) if len(sys.argv) > 6 else 400000
N_NVE = int(sys.argv[7]) if len(sys.argv) > 7 else 400000
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mai_port import Port, MASS
m = Port(root)
import jax, jax.numpy as jnp
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from epnn.preprocessing import create_batched_graph_list
from epnn.utils import get_nodes_from_graph
import jraph

name = os.path.basename(xyz)[:-4]; os.makedirs(outdir, exist_ok=True)
atoms0 = read(xyz); n = len(atoms0)
print("jax", jax.__version__, jax.devices(), f"{name}: {n} atoms, T {T} K, {R} replicas, NVT {N_NVT}, NVE {N_NVE} steps", flush=True)

# ---- energy and forces: the calculator's own function (ensemble mean of calc_potential_energy, summed over atoms)
calc = m.calculator(n_devices=1)
types = jnp.asarray([calc.symbol_map[s] for s in atoms0.symbols]); cell = jnp.zeros((3, 3), jnp.float32)
def energy(p):
    e = m.ensemble.apply(calc.params, p, types, cell, method=m.ensemble.calc_potential_energy)
    return e.sum() / m.ensemble.n_models
e_and_f = jax.value_and_grad(energy)
def ef(p):
    e, g = e_and_f(p); return e, -g

# ASE units: positions A, energies eV, masses amu, time in ASE units (1 fs = units.fs)
dt = 0.5 * units.fs
mass = jnp.asarray(atoms0.get_masses(), jnp.float32)[:, None]
kT = units.kB * T
gamma = 2e-2                                   # ASE Langevin friction, per ASE time unit (1_MD_calc_position.py)

def vv_step(state, _):
    p, v, f = state
    v = v + 0.5 * dt * f / mass; p = p + dt * v
    e, f = ef(p); v = v + 0.5 * dt * f / mass
    return (p, v, f), e

def remove_com(v):
    return v - (mass * v).sum(0) / mass.sum()

c1 = np.exp(-gamma * dt); c2 = np.sqrt(1 - c1 ** 2)
def baoab_step(state, key):
    p, v, f = state
    v = v + 0.5 * dt * f / mass; p = p + 0.5 * dt * v
    v = c1 * v + c2 * jnp.sqrt(kT / mass) * jax.random.normal(key, v.shape)
    v = remove_com(v); p = p + 0.5 * dt * v
    e, f = ef(p); v = v + 0.5 * dt * f / mass
    return (p, v, f), e

def ekin(v): return 0.5 * (mass * v ** 2).sum()

# ---- 1. jitted velocity Verlet vs ASE VelocityVerlet, 200 steps from identical positions and velocities
a = atoms0.copy(); a.calc = calc
MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(0))
p0 = jnp.asarray(a.positions, jnp.float32); v0 = jnp.asarray(a.get_velocities(), jnp.float32)
_, f0 = ef(p0)
run200 = jax.jit(lambda s: jax.lax.scan(vv_step, s, None, length=200))
(pj, vj, _), ej = run200((p0, v0, f0))
e_ase = [a.get_potential_energy() + a.get_kinetic_energy()]
md = VelocityVerlet(a, dt); md.attach(lambda: e_ase.append(a.get_potential_energy() + a.get_kinetic_energy()), interval=50)
md.run(200)
e_jit_tot = float(ej[-1] + ekin(vj))
print(f"[1] jitted VV vs ASE after 200 steps: max|dr| {np.abs(np.asarray(pj) - a.positions).max():.2e} A, "
      f"max|dv| {np.abs(np.asarray(vj) - a.get_velocities()).max():.2e} A/ase-t, "
      f"E_tot jit {e_jit_tot:.4f} vs ASE {e_ase[-1]:.4f} eV (ASE drift {e_ase[-1] - e_ase[0]:+.2e})", flush=True)

# ---- 2. production: R replicas
P, V = [], []
for k in range(R):
    a = atoms0.copy(); MaxwellBoltzmannDistribution(a, temperature_K=T, rng=np.random.default_rng(1000 + k))
    P.append(a.positions); V.append(a.get_velocities())
P = jnp.asarray(np.array(P), jnp.float32); V = jnp.asarray(np.array(V), jnp.float32)
F = jax.vmap(lambda p: ef(p)[1])(P)

CHUNK = 2000                                  # steps per jitted call (multiple of 2)
@jax.jit
def nvt_chunk(state, key):
    keys = jax.random.split(key, (CHUNK, R))
    def step(s, ks):
        return jax.vmap(baoab_step)(s, ks)
    s, e = jax.lax.scan(step, state, keys)
    return s, e[-1]

@jax.jit
def nve_chunk(state):
    def two(s, _):
        s, _e = jax.vmap(lambda st: vv_step(st, None))(s)
        s, e = jax.vmap(lambda st: vv_step(st, None))(s)
        return s, (s[0], e, jax.vmap(ekin)(s[1]))
    return jax.lax.scan(two, state, None, length=CHUNK // 2)

state = (P, V, F); key = jax.random.PRNGKey(20261110)
t0 = time.perf_counter(); nvt_T = []
for i in range(N_NVT // CHUNK):
    key, sub = jax.random.split(key); state, e = nvt_chunk(state, sub)
    if i % 20 == 0 or i == N_NVT // CHUNK - 1:
        tk = np.asarray(jax.vmap(ekin)(state[1])) / (1.5 * n * units.kB); nvt_T.append(tk)
        print(f"  NVT step {(i + 1) * CHUNK:7d}: T {np.round(tk, 1)} K  ({time.perf_counter() - t0:.0f} s)", flush=True)
t_nvt = time.perf_counter() - t0

frames, epot, ek = [], [], []
t0 = time.perf_counter()
for i in range(N_NVE // CHUNK):
    state, (pos, e, k_) = nve_chunk(state)
    frames.append(np.round(np.asarray(pos), 5)); epot.append(np.asarray(e)); ek.append(np.asarray(k_))
    if i % 40 == 0 or i == N_NVE // CHUNK - 1:
        print(f"  NVE step {(i + 1) * CHUNK:7d}: E_tot {np.round(epot[-1][-1] + ek[-1][-1], 4)} eV  ({time.perf_counter() - t0:.0f} s)", flush=True)
t_nve = time.perf_counter() - t0
frames = np.concatenate(frames); epot = np.concatenate(epot); ek = np.concatenate(ek)   # (n_frames, R, ...)
etot = epot + ek; temp = ek / (1.5 * n * units.kB)
print(f"[2] MD: NVT {t_nvt:.0f} s, NVE {t_nve:.0f} s for {R} replicas = "
      f"{(t_nvt + t_nve) / ((N_NVT + N_NVE) * 0.5e-3) :.2f} s per ps of all replicas together", flush=True)
for k in range(R):
    print(f"    replica {k}: NVE <T> {temp[:, k].mean():.1f} K, E_tot drift {etot[-1, k] - etot[0, k]:+.2e} eV, "
          f"std {etot[:, k].std():.2e} eV", flush=True)

# ---- 3. dipoles, batched as in 2_calc_dipole.py (COM-centred, cell zero), checked against Port.dipole
w = np.array([MASS[s] for s in atoms0.get_chemical_symbols()])
dtypes = jnp.array([m.d_map[s] for s in atoms0.get_chemical_symbols()])
B = 500
desc_batch = jax.jit(jax.vmap(m.d_desc))       # types and cells are arguments, not constants (XLA constant-folds those)
epnn_apply = jax.jit(m.epnn.apply)
def charges_batch(x, t):                          # x: (B, n, 3) COM-centred; t: (B, n); graph building is not traceable
    b = x.shape[0]; c0 = jnp.zeros((b, 3, 3), jnp.float32)
    d = desc_batch(x, t, c0)
    g = create_batched_graph_list(x, c0, d.reshape(b, n, -1), t, jnp.zeros((b, n, 1)),
                                  m.dp.r_cut, m.dp.r_switch, m.dp.node_state_dim, b)
    g = jraph.unpad_with_graphs(g[0])             # the builder pads edges to this batch's own count, so every batch
    e_pad = -(-int(g.n_edge.sum()) // 4096) * 4096 # has a new shape and recompiles: re-pad to a multiple of 4096
    g = jraph.pad_with_graphs(g, n_node=b * n + 1, n_edge=e_pad, n_graph=b + 1)
    return get_nodes_from_graph(epnn_apply(m.dp.params, g))[0].reshape(b, n)
tb = jnp.tile(dtypes, (B, 1))
def dipoles(fr):                                  # fr: (N, n, 3) float64 rounded positions
    pc = fr - (w[None, :, None] * fr).sum(1, keepdims=True) / w.sum()
    out = []
    for s in range(0, len(pc), B):
        x = np.zeros((B, n, 3), np.float32); b = min(B, len(pc) - s); x[:b] = pc[s:s + b]   # pad the last batch
        q = np.asarray(charges_batch(jnp.asarray(x), tb))[:b]
        out.append((q[:, :, None] * x[:b]).sum(1))
    return np.concatenate(out)
chk = frames[[0, len(frames) // 2, -1], -1]; mu_b = dipoles(chk)   # start, middle and end of NVE
mu_1 = np.array([m.dipole(atoms0.__class__(atoms0.symbols, positions=c))[0] for c in chk])
print(f"[3] batched vs per-frame dipole (NVE start, middle, end): max|d| {np.abs(mu_b - mu_1).max():.2e} e*A", flush=True)

t0 = time.perf_counter(); dipoles(frames[:B, 0]); t1 = time.perf_counter(); dipoles(frames[:B, 0])
print(f"[3] one batch of {B} frames: {time.perf_counter() - t1:.2f} s after compile ({t1 - t0:.0f} s first call)", flush=True)
t0 = time.perf_counter()
for k in range(R):
    mu = dipoles(frames[:, k].astype(np.float64))
    tag = f"{name}_s{k}"; d = f"{outdir}/run_s{k}"
    for sub in ("code", "inputs/XYZ", "intermediate_results/Dipole_txt_mass", "outputs"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)
    open(f"{d}/inputs/XYZ/{tag}.xyz", "w").write(open(xyz).read())
    np.savetxt(f"{d}/intermediate_results/Dipole_txt_mass/VelocityVerlet_{T}K_{tag}.xyz_dipole.txt",
               np.column_stack((np.arange(len(mu)), mu)), fmt="%d %.10f %.10f %.10f", header="time dipole_x dipole_y dipole_z")
    np.savez(f"{outdir}/md_s{k}.npz", etot=etot[:, k], temp=temp[:, k], nvt_T=np.array(nvt_T)[:, k])
print(f"[3] dipoles for {R} x {len(frames)} frames in {time.perf_counter() - t0:.0f} s; written under {outdir}/run_s*", flush=True)
