# T0 week-4 TorchSim pilot (plan 4.3), run early: 100 PAHs, 1 ps Langevin NVT at 600 K, 0.5 fs, gamma 0.01/fs, float32, no TF32.
import time, numpy as np, torch
torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = False
import os; assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" not in os.environ
import torch_sim as ts
from torch_sim.units import MetalUnits as MU
from torch_sim.integrators import nvt_langevin_init, nvt_langevin_step
from torch_sim.quantities import calc_temperature
from mace.calculators import mace_off
from mace.calculators.mace_torchsim import MaceTorchSimModel
from ase import Atoms
dev = torch.device("cuda")

def pah(rings):  # rings: axial hex coords; returns ideal planar PAH (C-C 1.40, C-H 1.09)
    a = 1.40; c = []
    for q, r in rings:
        cx, cy = a*np.sqrt(3)*(q + r/2), a*1.5*r
        for k in range(6):
            t = np.pi/6 + k*np.pi/3; c.append((cx + a*np.cos(t), cy + a*np.sin(t)))
    c = np.unique(np.round(np.array(c), 3), axis=0)
    d = np.linalg.norm(c[:, None] - c[None], axis=-1); nb = (d < 1.5) & (d > 0.1)
    h = []
    for i in np.where(nb.sum(1) == 2)[0]:
        v = 2*c[i] - c[nb[i]].sum(0); h.append(c[i] + 1.09*v/np.linalg.norm(v))
    xyz = np.vstack([c, h]); xyz = np.c_[xyz, np.zeros(len(xyz))]
    return Atoms("C%dH%d" % (len(c), len(h)), positions=xyz - xyz.mean(0))
ring7 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
SPECIES = {"benzene": [(0,0)], "naphthalene": [(0,0),(1,0)], "pyrene": [(0,0),(1,0),(0,1),(1,-1)],
           "coronene": ring7, "ovalene": ring7 + [(2,-1),(1,1),(2,0)],
           "circumcoronene": ring7 + [(2,0),(1,1),(0,2),(-1,2),(-2,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-2),(2,-1)]}
mols = {k: pah(v) for k, v in SPECIES.items()}
print({k: m.get_chemical_formula() for k, m in mols.items()})
names = [list(mols)[i % len(mols)] for i in range(100)]
batch = [mols[n].copy() for n in names]   # all at the origin: cross-system edges would be obvious

raw = mace_off(model="small", default_dtype="float32", device="cuda", return_raw_model=True)
model = MaceTorchSimModel(model=raw, device=dev, dtype=torch.float32, compute_forces=True, compute_stress=False)
print("model dtype", model.dtype, "stress", model.compute_stress)

T, DT_FS, GAMMA_FS, NSTEP = 600.0, 0.5, 0.01, 2000
kT = torch.tensor(T * MU.temperature, device=dev, dtype=torch.float32)        # K -> eV
dt = torch.tensor(DT_FS * 1e-3 * MU.time, device=dev, dtype=torch.float32)    # ps -> internal
gamma = torch.tensor(GAMMA_FS * 1e3 / MU.time, device=dev, dtype=torch.float32)  # 1/ps -> internal
print(f"internal: kT {kT.item():.5f} eV, dt {dt.item():.5f}, gamma {gamma.item():.5f} (gamma*dt = {(gamma*dt).item():.5f}, expect 0.005)")

state = ts.io.atoms_to_state(batch, device=dev, dtype=torch.float32); state.rng = 1234
torch.cuda.synchronize(); t0 = time.perf_counter()
state = nvt_langevin_init(state, model, kT=kT)
torch.cuda.synchronize(); print(f"init {time.perf_counter()-t0:.2f} s")
for _ in range(3): model(state)
torch.cuda.synchronize(); t1 = time.perf_counter()
for _ in range(20): model(state)
torch.cuda.synchronize(); print(f"model forward alone (100 mols, {state.n_atoms} atoms): {(time.perf_counter()-t1)/20*1e3:.1f} ms")
sid = state.system_idx; nat = torch.bincount(sid)
def temps(s):
    return calc_temperature(masses=s.masses, momenta=s.momenta, system_idx=s.system_idx)  # returns kelvin
v0, Ts, Tlog = [], [], []
torch.cuda.synchronize(); t0 = time.perf_counter()
for k in range(NSTEP):
    state = nvt_langevin_step(state, model, dt=dt, kT=kT, gamma=gamma)
    v0.append((state.momenta[sid == 0] / state.masses[sid == 0, None]).detach())  # benzene, for the VDOS
    if k % 100 == 99: Tlog.append(temps(state).detach())
torch.cuda.synchronize(); wall = time.perf_counter() - t0
print(f"1 ps x 100 molecules: {wall:.1f} s wall  ->  {wall/(100*1.0):.4f} s per molecule-ps, "
      f"{86400/wall*1e-3:.2f} ns/day per trajectory, peak GPU mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

Tl = torch.stack(Tlog).cpu().numpy()   # (20, 100), dof = 3N, COM unconstrained
late = Tl[10:].mean(0)
print(f"T (last 0.5 ps, per molecule): mean {late.mean():.1f} K, by species:",
      {n: round(float(late[[i for i in range(100) if names[i]==n]].mean()), 1) for n in mols})
assert torch.isfinite(state.positions).all() and torch.isfinite(state.energy).all()

# No cross-molecule edges: all 100 molecules overlap in space, so batched forces at the final state
# must equal isolated single-molecule ASE forces.
calc = mace_off(model="small", default_dtype="float32", device="cuda")
pos = state.positions.detach().cpu().numpy(); F = model(state)["forces"].detach().cpu().numpy()
dmax = 0.0
for i in range(100):
    m = mols[names[i]].copy(); m.positions = pos[(sid == i).cpu().numpy()]; m.calc = calc
    dmax = max(dmax, np.abs(m.get_forces() - F[(sid == i).cpu().numpy()]).max())
print(f"final-state batched vs isolated ASE: max|dF| = {dmax:.2e} eV/A (gate 1e-3)")

# Time units: C-H stretch from the benzene velocity autocorrelation should sit near 3050 cm^-1.
v = torch.stack(v0).cpu().numpy()   # (NSTEP, 12, 3) internal units; only the frequency axis matters
spec = np.abs(np.fft.rfft(v - v.mean(0), axis=0))**2; spec = spec.sum((1, 2))
freq_cm = np.fft.rfftfreq(NSTEP, d=DT_FS * 1e-15) / 2.99792458e10
hi = freq_cm > 2500; print(f"benzene VDOS peak above 2500 cm^-1: {freq_cm[hi][np.argmax(spec[hi])]:.0f} cm^-1; "
      f"below 2000: {freq_cm[~hi & (freq_cm>200)][np.argmax(spec[~hi & (freq_cm>200)])]:.0f} cm^-1")
