# Figure for reports/nc_sr_v4 (section 2.1): the DF3 template under the three continua, and the library fit A.
#   python slice/figure_df3.py <out png> [orion_gap.npz, default /var/tmp/$USER-pah/sr_v4/gap/orion_gap.npz]   (numpy, matplotlib)
import numpy as np, matplotlib, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import os; d = np.load(sys.argv[2] if len(sys.argv) > 2 else f"/var/tmp/{os.environ["USER"]}-pah/sr_v4/gap/orion_gap.npz"); w = d["wav"]; t = "DF3"
F, cP, cS, cL = d[t+"_flux"], d[t+"_cont_pahfit"], d[t+"_cont_spl"], d[t+"_cont_lin"]
lib, con = d[t+"_A_lib"], d[t+"_A_cont"]
I = lambda y, lo, hi: np.trapezoid((y / w**2)[(w>=lo)&(w<=hi)], w[(w>=lo)&(w<=hi)])
r = {n: I(F-c, 7.0, 8.15) / I(F-c, 10.9, 11.7) for n, c in (("PAHFIT", cP), ("spline", cS), ("straight", cL))}
m = (w >= 5.2) & (w <= 15.5)
fig, ax = plt.subplots(3, 1, figsize=(10, 10.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 1.3]))
a = ax[0]
a.fill_between(w[m], cP[m], np.minimum(cS, F)[m], where=(w[m] < 10.0), color="#edae49", alpha=0.45, lw=0, label="flux the two methods assign differently")
a.plot(w[m], F[m], color="k", lw=0.8, label="Orion Bar DF3 template (gas lines removed)")
a.plot(w[m], cP[m], color="#d1495b", lw=1.8, label="continuum: PAHFIT")
a.plot(w[m], cS[m], color="#1f6feb", lw=1.8, label="continuum: local, natural spline")
a.plot(w[m], cL[m], color="#1f6feb", lw=1.2, ls="--", label="continuum: local, straight segments")
for lo, hi in ((5.45,5.55),(5.88,5.95),(9.55,9.9),(9.9,10.1),(14.6,14.9),(15.0,15.4)):
    a.axvspan(lo, hi, color="0.85", lw=0)
a.text(9.83, 300, "anchors", ha="center", fontsize=8, color="0.35"); a.text(5.7, 6900, "anchors", ha="center", fontsize=8, color="0.35")
a.annotate("the spline bulges here:\nno anchor for 4 µm", xy=(7.9, float(np.interp(7.9, w, cS))), xytext=(8.6, 4600), fontsize=9, arrowprops=dict(arrowstyle="->", color="#1f6feb"), color="#1f6feb")
a.set_ylim(0, 7600); a.set_ylabel("surface brightness (MJy/sr)"); a.legend(frameon=False, fontsize=8.5, loc="upper right")
a.set_title(f"1. Same spectrum, different continuum.  7.7/11.2 band ratio: PAHFIT {r['PAHFIT']:.1f}, straight {r['straight']:.1f}, spline {r['spline']:.1f}"
            "\n→ cation fraction 0.75 (PAHFIT) against 0.38 (local spline)", fontsize=10.5, loc="left")
a = ax[1]
a.plot(w[m], F[m], color="k", lw=0.8, label="template")
a.plot(w[m], (lib + con)[m], color="#2a9d8f", lw=1.3, label="best mixture of library spectra + continuum")
a.plot(w[m], con[m], color="#2a9d8f", lw=1.0, ls=":", label="its continuum part")
a.set_ylim(0, 7600); a.set_ylabel("surface brightness (MJy/sr)"); a.legend(frameon=False, fontsize=8.5, loc="upper right")
a.set_title("2. The library can reproduce the template (rms 9 %), so no emission is plainly missing", fontsize=10.5, loc="left")
a = ax[2]
a.axhline(0, color="0.4", lw=0.6); a.plot(w[m], ((F - lib - con) / F)[m], color="#2a9d8f", lw=0.8)
a.set_ylim(-0.35, 0.6); a.set_ylabel("residual / template"); a.set_xlabel("wavelength (µm)")
a.annotate("5.25 and 5.7 µm bands:\nnot in the library", xy=(5.7, 0.33), xytext=(6.6, 0.38), fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.3"))
a.annotate("6.2 µm band profile", xy=(6.2, 0.2), xytext=(8.0, 0.3), fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.3"))
for x in ax: x.spines[["top", "right"]].set_visible(False); x.set_xlim(5.2, 15.5)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=130); print(r)
