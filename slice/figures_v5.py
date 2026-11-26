# Figures of reports/nc_sr_v5: the Orion continuum checks.
#   python slice/figures_v5.py <orion_cont dir of slice/orion_cont.py> <Orion template file (A75 sptemp.tsv)> <out dir>
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as M  # noqa: E402


def main():
    d, tpl, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    Z = np.load(os.path.join(d, "orion_cont_curves.npz"))
    # 1. PAHFIT continuum (classic and with more freedom) against the monotone local continuum, 5-13 um
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for a, t in zip(ax.T, ("APDR", "DF3")):
        g = lambda k, p="classic": Z[f"{t}@{p}|{k}"]  # noqa: E731
        w = g("wav"); m = (w > 5.2) & (w < 13.2); f = g("flux")
        a[0].plot(w[m], f[m], "k", lw=0.7, label="template (running median)")
        a[0].plot(w[m], g("model")[m], color="C1", lw=0.7, label="PAHFIT classic, model")
        a[0].plot(w[m], g("cont")[m], color="C1", ls="--", label="PAHFIT classic, continuum")
        a[0].plot(w[m], g("cont", "hotplat")[m], color="C3", ls=":", lw=2, label="PAHFIT + hot blackbodies + plateau profiles, continuum")
        a[0].plot(w[m], g("pch")[m], color="C0", ls="--", label="monotone local continuum (Lpch)")
        for lo, hi in M.LANCH[:4]:
            a[0].axvspan(lo, hi, color="C0", alpha=0.15, lw=0)
        a[0].set_title(f"{t}: continuum under the 6-9 µm bands", fontsize=10); a[0].set_ylim(0, 1.05 * f[(w > 7) & (w < 9)].max())
        a[1].plot(w[m], g("bands")[m], color="C1", lw=0.8, label="PAHFIT band profiles (Drude)")
        a[1].plot(w[m], (f - g("pch"))[m], color="C0", lw=0.8, label="template minus the local continuum")
        a[1].plot(w[m], (f - g("model"))[m], color="0.5", lw=0.6, label="template minus the PAHFIT model")
        for lo, hi in M.LANCH[:4]:
            a[1].axvspan(lo, hi, color="C0", alpha=0.15, lw=0)
        a[1].set_ylim(-0.12 * f[(w > 7) & (w < 9)].max(), 1.0 * f[(w > 7) & (w < 9)].max()); a[1].axhline(0, color="k", lw=0.4)
        a[1].set_xlabel("wavelength (µm)"); a[1].set_title("what each method calls band emission (shaded: local anchors)", fontsize=10)
    ax[0, 0].set_ylabel("MJy/sr"); ax[1, 0].set_ylabel("MJy/sr"); ax[0, 0].legend(fontsize=7, loc="upper right"); ax[1, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_orion_cont.png"), dpi=130); plt.close(fig)
    # 2. the 17 um complex above a chord (Lpch) and above the curved continuum
    wav, T = M.read_templates(tpl); fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, t in zip(ax, ("APDR", "DF3", "DF1")):
        f = median_filter(T[t], 11, mode="nearest"); m = (wav > 14.2) & (wav < 20.0)
        x = np.array([0.5 * (lo + hi) for lo, hi in M.LANCH]); y = np.array([np.median(f[(wav >= lo) & (wav <= hi)]) for lo, hi in M.LANCH])
        pch = PchipInterpolator(x, y)(wav[m]); k = x > 14
        crv = np.exp(np.polyval(np.polyfit(np.log(x[k]), np.log(y[k]), 2), np.log(wav[m])))
        a.plot(wav[m], f[m] / 1e3, "k", lw=0.7, label="template"); a.plot(wav[m], pch / 1e3, "C0--", label="monotone (Lpch)")
        a.plot(wav[m], crv / 1e3, "C3", label="parabola in log-log (Lcrv)"); a.plot(x[k], y[k] / 1e3, "C3o", ms=4)
        a.axvspan(*M.WINDOWS["17"], color="0.85", lw=0); a.set_ylim(0.8 * f[m].min() / 1e3, 1.1 * np.percentile(f[m], 99) / 1e3)
        a.set_title(t, fontsize=10); a.set_xlabel("wavelength (µm)")
    ax[0].set_ylabel("10³ MJy/sr"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_17um_continuum.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
