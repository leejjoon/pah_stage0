# Plan 4.2 gate: forces < 1e-3 eV/A and energies < 1e-4 eV/atom, per cuEquivariance combination.
# Energies are also compared with a float64 PyTorch reference, to show the float32 rounding floor.
import numpy as np, csv
from ase.io import read
sp = list(csv.DictReader(open("/t/eq_species.csv")))
fr = read("/t/eq_frames.xyz", ":"); nat = np.array([len(a) for a in fr]); sidx = np.array([a.info["species"] for a in fr])
T = {c: np.load(f"/t/eq_torch_cueq{c}.npz", allow_pickle=True) for c in (0, 1)}
J = {c: np.load(f"/t/eq_jax{'_cueq' if c else ''}.npz", allow_pickle=True) for c in (0, 1)}
def maxdf(A, B): return np.array([np.abs(np.asarray(a, float) - np.asarray(b, float)).max() for a, b in zip(A, B)])
for jc in (0, 1):
    for tc in (0, 1):
        dF = maxdf(J[jc]["forces"], T[tc]["forces"])
        dE = np.abs(J[jc]["energy"].astype(float) - T[tc]["energy"].astype(float)) / nat
        i = int(dF.argmax())
        print(f"jax cueq={jc} vs torch cueq={tc}: max|dF| {dF.max():.2e} eV/A ({sp[sidx[i]]['formula']}), median {np.median(dF):.1e}; "
              f"max|dE|/atom {dE.max():.1e} eV  forces {'PASS' if dF.max() < 1e-3 else 'FAIL'}, energy {'PASS' if dE.max() < 1e-4 else 'FAIL'}")
dF = maxdf(J[0]["forces"], T[0]["forces"])
print("max|dF| by size range (no cueq):", {r: f"{dF[np.isin(sidx, [int(s['idx']) for s in sp if s['range'] == r])].max():.1e}" for r in dict.fromkeys(s["range"] for s in sp)})
F = [np.abs(np.asarray(f, float)).max() for f in T[0]["forces"]]
print(f"max|F| per frame: {min(F):.2f}-{max(F):.2f} eV/A")
E64 = np.load("/t/eq_torch_f64_energy.npy")
for name, E in (("torch f32", T[0]["energy"]), ("torch f32 cueq", T[1]["energy"]), ("jax f32", J[0]["energy"]), ("jax f32 cueq", J[1]["energy"])):
    print(f"{name:15s} vs torch float64: max|dE|/atom {(np.abs(E.astype(float) - E64) / nat).max():.1e} eV")
sp32 = np.abs(np.spacing(E64.astype(np.float32)))
print(f"float32 spacing at these totals: up to {sp32.max():.4f} eV, i.e. {(sp32 / nat).max():.1e} eV/atom for one rounding step")
