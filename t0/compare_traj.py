"""Trajectory checks across stacks (plan 4.4 Correctness): means and spreads of temperature and potential energy per
trajectory, and the VDOS peaks of trajectory 0, for the npz files written by md_jax.py / md_torch.py.
Different RNG streams, so trajectories are not identical; statistics should agree within their noise.
Usage: compare_traj.py name=file.npz [name=file.npz ...]"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

runs = dict(a.split("=", 1) for a in sys.argv[1:])
D = {k: np.load(v) for k, v in runs.items()}

def series(d, key, k):
    if key in d.files: return d[key][:, k] if d[key].ndim > 1 else (d[key] if k == 0 else None)
    if f"{key}_{k}" in d.files: return d[f"{key}_{k}"]
    return None

print(f"{'run':14s} {'T mean':>8s} {'T sd':>6s}   (all trajectories, production)")
for n, d in D.items():
    T = d["T"] if "T" in d.files else np.array([d[f] for f in d.files if f.startswith("T_")]).T
    print(f"{n:14s} {T.mean():8.1f} {T.std():6.1f}")

names = list(D); ref = names[0]
nt = max((D[n]["T"].shape[1] if "T" in D[n].files and D[n]["T"].ndim > 1 else 1) for n in names)
per_traj = lambda d: "epot" in d.files and d["epot"].ndim > 1
print(f"\nper-trajectory potential energy (mean +- sd over the run, eV), first {nt} trajectories; diff = mean - {ref}, in units of the sd")
for k in range(nt):
    cells = []
    for n in names:
        e = series(D[n], "epot", k) if (per_traj(D[n]) or f"epot_{k}" in D[n].files) else None
        if e is None: continue
        e = e.astype(np.float64); e0 = series(D[ref], "epot", k).astype(np.float64)
        cells.append(f"{n}: {e.mean():.3f}+-{e.std():.3f}" + ("" if n == ref else f" ({(e.mean() - e0.mean()) / e0.std():+.2f} sd)"))
    print(f"  traj {k:2d}: " + " | ".join(cells))

for n in names:                                       # one-Atoms runs record only the batch total
    if "epot" in D[n].files and D[n]["epot"].ndim == 1 and per_traj(D[ref]):
        e, e0 = D[n]["epot"].astype(np.float64), D[ref]["epot"].astype(np.float64).sum(1)
        print(f"  batch total: {n} {e.mean():.3f}+-{e.std():.3f} vs {ref} {e0.mean():.3f}+-{e0.std():.3f} ({(e.mean() - e0.mean()) / e0.std():+.2f} sd)")
print("\nVDOS of trajectory 0 (peaks)")
for n, d in D.items():
    v = d["vel"] if "vel" in d.files else (d["vel_0"] if "vel_0" in d.files else None)
    if v is None or len(v) == 0: continue
    nu, p = C.vdos(v.astype(np.float64), C.DT_FS)
    hi, lo = nu > 2500, (nu > 500) & (nu < 2000)
    print(f"  {n:14s} C-H stretch {nu[hi][p[hi].argmax()]:.0f} cm^-1, strongest 500-2000 {nu[lo][p[lo].argmax()]:.0f} cm^-1  ({len(v)} frames)")
