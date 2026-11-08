# Plan 4.2 gates (revised 2026-11-08) on the 120 frames:
#   1. float32 forces            < 1e-3 eV/A    every cuEquivariance combination
#   2. float32 interaction energy < 1e-4 eV/atom every cuEquivariance combination
#   3. float64 total energy       < 1e-6 eV/atom x64-converted bundle, cueq off and on
# Float32 totals are reported, not gated, with the float32 spacing per frame, each stack's own f32-vs-f64
# deviation, signed residuals, and a decomposition check of the float64 residual.
import csv, numpy as np
from ase.io import read
fr = read("/t/eq_frames.xyz", ":")
nat = np.array([len(a) for a in fr]); sidx = np.array([a.info["species"] for a in fr])
sp = list(csv.DictReader(open("/t/eq_species.csv")))
T = {(d, c): np.load(f"/t/torch_{d}_cueq{c}.npz", allow_pickle=True) for d in ("float32", "float64") for c in (0, 1)}
import os
SUF = {0: "", 1: "_cueq", 2: "_cueqf"}   # 2 = cueq backend with conv_fusion (uniform_1d kernel); included if present
JV = [c for c in SUF if all(os.path.exists(f"/t/jax_{d}{SUF[c]}.npz") for d in ("float32", "float64"))]
J = {(d, c): np.load(f"/t/jax_{d}{SUF[c]}.npz", allow_pickle=True) for d in ("float32", "float64") for c in JV}
print("JAX variants: cueq=0 default bundle, 1 backend cueq, 2 backend cueq + conv_fusion; present:", JV)
f = lambda x: np.asarray(x, np.float64)
def maxdf(A, B): return np.array([np.abs(f(a) - f(b)).max() for a, b in zip(A, B)])
res = {}
def gate(name, val, tol, unit):
    ok = val < tol; res[name] = ok
    print(f"  {name:52s} max {val:.2e} {unit}  (gate {tol:.0e})  {'PASS' if ok else 'FAIL'}")

print("== dtypes of saved outputs")
for k, v in {**{('torch',) + k: T[k] for k in T}, **{('jax',) + k: J[k] for k in J}}.items():
    print(f"  {k}: energy {np.asarray(v['energy']).dtype}, interaction_energy {np.asarray(v['interaction_energy']).dtype}")

print("== JAX bundle variants against the default bundle (same model; differences are summation order)")
for d in ("float32", "float64"):
  for c in JV[1:]:
    a, b = J[(d, 0)], J[(d, c)]
    print(f"  jax {d} cueq=0 vs {c}: max|dE| {np.abs(f(a['energy']) - f(b['energy'])).max():.1e} eV, max|dF| {maxdf(a['forces'], b['forces']).max():.1e} eV/A, "
          f"bit-identical energies: {np.array_equal(a['energy'], b['energy'])}")

print("== gate 1 and 2: float32, every cuEquivariance combination")
for jc in JV:
    for tc in (0, 1):
        j, t = J[("float32", jc)], T[("float32", tc)]
        gate(f"forces f32, jax cueq={jc} vs torch cueq={tc}", maxdf(j["forces"], t["forces"]).max(), 1e-3, "eV/A")
        gate(f"interaction energy f32, jax cueq={jc} vs torch cueq={tc}",
             (np.abs(f(j["interaction_energy"]) - f(t["interaction_energy"])) / nat).max(), 1e-4, "eV/atom")

print("== gate 3: float64 totals, x64-converted bundle")
for jc in JV:
    for tc in (0, 1):
        j, t = J[("float64", jc)], T[("float64", tc)]
        gate(f"total energy f64, jax cueq={jc} vs torch cueq={tc}", (np.abs(f(j["energy"]) - f(t["energy"])) / nat).max(), 1e-6, "eV/atom")

print("== reported, not gated")
for jc in JV:
    for tc in (0, 1):
        dE = np.abs(f(J[("float32", jc)]["energy"]) - f(T[("float32", tc)]["energy"])) / nat
        print(f"  total energy f32, jax cueq={jc} vs torch cueq={tc}: max {dE.max():.1e} eV/atom")
E64 = f(T[("float64", 0)]["energy"])
spc = np.abs(np.spacing(E64.astype(np.float32)).astype(np.float64)) / nat
print(f"  float32 spacing / atoms per frame: {spc.min():.1e} to {spc.max():.1e} eV/atom (median {np.median(spc):.1e})")
for name, a, b in (("PyTorch f32 vs PyTorch f64", T[("float32", 0)], T[("float64", 0)]),
                   ("PyTorch f32 vs PyTorch f64 (cueq)", T[("float32", 1)], T[("float64", 1)]),
                   ("JAX f32 vs JAX f64", J[("float32", 0)], J[("float64", 0)]),
                   ("JAX f32 vs JAX f64 (cueq bundle)", J[("float32", 1)], J[("float64", 1)])):
    print(f"  {name:36s} total {(np.abs(f(a['energy']) - f(b['energy'])) / nat).max():.1e} eV/atom, "
          f"interaction {(np.abs(f(a['interaction_energy']) - f(b['interaction_energy'])) / nat).max():.1e} eV/atom, "
          f"forces {maxdf(a['forces'], b['forces']).max():.1e} eV/A")
j, t = J[("float64", 0)], T[("float64", 0)]
print(f"  f64 forces jax vs torch: max {maxdf(j['forces'], t['forces']).max():.1e} eV/A; "
      f"f64 interaction energy: max {(np.abs(f(j['interaction_energy']) - f(t['interaction_energy'])) / nat).max():.1e} eV/atom")

print("== signed residuals per species (JAX - PyTorch, cueq off/off): mean and spread over the 6 frames, eV")
r64 = f(j["energy"]) - f(t["energy"])
r32t = f(J[("float32", 0)]["energy"]) - f(T[("float32", 0)]["energy"])
r32i = f(J[("float32", 0)]["interaction_energy"]) - f(T[("float32", 0)]["interaction_energy"])
r64i = f(j["interaction_energy"]) - f(t["interaction_energy"])
print(f"  {'species':10s} {'atoms':>5s} {'f64 total':>20s} {'f64 E0 part':>12s} {'f32 total':>20s} {'f32 interaction':>20s}")
for s in sp:
    m = sidx == int(s["idx"])
    e0part = (r64 - r64i)[m]          # total - interaction = the E0 sum's contribution to the residual
    print(f"  {s['formula']:10s} {nat[m][0]:5d} {r64[m].mean():+.2e} ±{r64[m].std():.1e} {np.abs(e0part).max():12.1e} "
          f"{r32t[m].mean():+.2e} ±{r32t[m].std():.1e} {r32i[m].mean():+.2e} ±{r32i[m].std():.1e}")
print("  'f64 E0 part' = max |residual(total) - residual(interaction)| = the E0-sum contribution (zero if E0 match: sum_Z n_Z dE0_Z)")
print("== summary:", "ALL GATES PASS" if all(res.values()) else "FAILED: " + ", ".join(k for k, v in res.items() if not v))
