# Step 2b: compare the port's spectra (one per replica, from the authors' 3_calc_IR.py) with the published one.
# One trajectory of a small molecule is noisy, so compare per band window: peak and centroid position, and the
# band's share of the total 300-3800 cm^-1 intensity, against the scatter of the replicas. Not pointwise.
# Usage: python mai_spectrum_compare.py <published spectrum txt> <png or -> <port spectrum txt> [...]
import sys
import numpy as np
trap = getattr(np, "trapezoid", None) or np.trapz      # NumPy 2 renamed trapz

pub, png, ours = sys.argv[1], sys.argv[2], sys.argv[3:]
WINDOWS = [(440, 520), (740, 820), (930, 985), (985, 1030), (1100, 1180), (1230, 1290), (1350, 1420),
           (1480, 1560), (1570, 1640), (2980, 3150)]

def bands(f):
    x, y = np.loadtxt(f, unpack=True); tot = trap(y, x); out = []
    for lo, hi in WINDOWS:
        s = (x >= lo) & (x < hi); xs, ys = x[s], y[s]; a = trap(ys, xs)
        out.append((xs[ys.argmax()], (xs * ys).sum() / ys.sum() if ys.sum() > 0 else np.nan, a / tot))
    return np.array(out), (x, y)

P, pxy = bands(pub); O = [bands(f) for f in ours]; Ob = np.array([o[0] for o in O]); N = len(O)
# replica-mean spectrum (all replicas share one frequency grid)
xm = O[0][1][0]; ym = np.mean([o[1][1] for o in O], axis=0); ym /= ym.max()
Mb = []
tot = trap(ym, xm)
for lo, hi in WINDOWS:
    s_ = (xm >= lo) & (xm < hi); xs, ys = xm[s_], ym[s_]
    Mb.append(((xs * ys).sum() / ys.sum(), trap(ys, xs) / tot))
print(f"{N} replicas. Share = band's fraction of the 300-3800 cm-1 intensity. Rank = replicas with a smaller share than"
      f" the published one (outside the range by chance: 2/(N+1) = {2 / (N + 1):.2f} per window).")
print(f"{'window':>11s} | {'published':>18s} | {'replica-mean spectrum':>21s} | {'replicas: share median [min..max]':>34s} rank")
print(f"{'cm-1':>11s} | {'centroid':>8s} {'share':>9s} | {'centroid':>10s} {'share':>10s} |")
for i, (lo, hi) in enumerate(WINDOWS):
    pk, ce, sh = P[i]; osh = Ob[:, i, 2]
    print(f"{lo:5d}-{hi:<5d} | {ce:8.1f} {sh:9.4f} | {Mb[i][0]:10.1f} {Mb[i][1]:10.4f} | "
          f"{np.median(osh):8.4f} [{osh.min():.4f}..{osh.max():.4f}]   {int((osh < sh).sum()):3d}/{N}")
strong = [i for i in range(len(WINDOWS)) if P[i, 2] > 0.05]
print("strong bands (published share > 5%): centroid offset, published - replica-mean spectrum:",
      ", ".join(f"{WINDOWS[i][0]}-{WINDOWS[i][1]}: {P[i, 1] - Mb[i][0]:+.1f} cm-1" for i in strong))
if png != "-":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    for a, xl in zip(ax, ((300, 1700), (2900, 3200))):
        a.plot(*pxy, "k", lw=1.5, label="published (Mai+2025, 300 K)")
        a.plot(xm, ym, color="C3", lw=1.0, label=f"port, mean of {N} replicas")
        a.set_xlim(*xl); a.set_xlabel("wavenumber (cm$^{-1}$, scaled 0.9578)"); a.set_ylabel("normalised intensity")
    ax[0].legend(); fig.tight_layout(); fig.savefig(png, dpi=120); print("figure:", png)
