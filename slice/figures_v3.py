# Figures for the v3 report (neutral + cation).
#   python slice/figures_v3.py <regress_v2 out dir> <fig dir>
# fig_sets.png, fig_pred.png   as figures_v2.py, with the cation fraction as a third target
# fig_b4.png      Gate B4 step b: change of the held-out RMSE and mean shift of each predictor when only the cation
#                 rules are changed, per target; local integrals, MIRI only and with 3.3 um
# fig_orion.png   the five Orion Bar templates: predicted cation fraction, size and photon energy per observable set
#                 without the 17 um ratio
#                 (random forest; bars = scatter of the Monte Carlo refits; hollow = outside the training domain)
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figures_v2 as F2  # noqa: E402

VAR = [("harmq", "cations\nharmonic"), ("cc17", "CC strength\n−1.7"), ("cc30", "CC strength\n−3.0"), ("shp30", "shifts\n+30 %"), ("shm30", "shifts\n−30 %"),
       ("wdp30", "widths\n+30 %"), ("wdm30", "widths\n−30 %")]
PRED = [("rf", "random forest", "#1f6feb"), ("formula_compact", "formula ≤ 7", "#d1495b"), ("formula_full", "formula ≤ 12", "#edae49")]
TEMPL = ["HII", "APDR", "DF1", "DF2", "DF3"]


def main(reg, out):
    F2.main(reg, out)
    res = json.load(open(os.path.join(reg, "results.json"))); S = res["sets"]
    sets = [s for s in ("L", "L33") if s in S]; tg = list(S[sets[0]]["targets"])
    fig, axs = plt.subplots(2 * len(sets), len(tg), figsize=(5.6 * len(tg), 2.6 * 2 * len(sets)), squeeze=False, sharex=True)
    for i, s in enumerate(sets):
        for j, t in enumerate(tg):
            T = S[s]["targets"][t]; va = [(v, lab) for v, lab in VAR if f"control_{v}" in T]
            for row, key, yl in ((2 * i, "d", "change of RMSE"), (2 * i + 1, "shift", "mean shift")):
                ax = axs[row][j]; w = 0.26
                for k, (pn, lab, col) in enumerate(PRED):
                    ys = [(T[f"control_{v}"][pn]["other"] - T[f"control_{v}"][pn]["main"]) if key == "d" else T[f"control_{v}"][pn]["shift"]
                          for v, _ in va if pn in T[f"control_{v}"]]
                    if ys:
                        ax.bar(np.arange(len(ys)) + (k - 1) * w, ys, w * 0.92, color=col, label=lab)
                ax.axhline(0, color="0.3", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
                base = T["rf"]["test"]; ax.axhline(base, color="0.5", lw=0.6, ls=":"); ax.axhline(-base, color="0.5", lw=0.6, ls=":")
                ax.set_ylabel(f"{yl}\n{F2.UNIT.get(t, t)}", fontsize=8)
                if row == 0:
                    ax.set_title(F2.UNIT.get(t, t), fontsize=10)
                ax.text(0.01, 0.97, F2.NAME_ALL[s].replace("\n", ", ") + " (dotted: ± forest test RMSE)", transform=ax.transAxes, va="top", fontsize=7)
                ax.set_xticks(range(len(va))); ax.set_xticklabels([lab for _, lab in va], fontsize=7)
    axs[0][0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_b4.png"), dpi=130); plt.close(fig)
    osets = [s for s in ("Pn", "P33n", "Ln", "L33n") if s in S and "orion" in S[s]["targets"][tg[0]]]
    if osets:
        fig, axs = plt.subplots(1, len(tg), figsize=(5.2 * len(tg), 3.8), squeeze=False)
        col = {"Pn": "#9aa0a6", "P33n": "#5f6368", "Ln": "#7fb3ff", "L33n": "#1f6feb"}
        for ax, t in zip(axs[0], tg):
            for k, s in enumerate(osets):
                O = S[s]["targets"][t]["orion"]; x = np.arange(len(TEMPL)) + (k - (len(osets) - 1) / 2) * 0.18
                y = np.array([O[tn]["rf"] for tn in TEMPL]); e = np.array([O[tn]["rf_mc_sd"] or 0 for tn in TEMPL])
                ins = np.array([not O[tn]["extrapolation"] for tn in TEMPL])
                ax.errorbar(x, y, e, fmt="none", ecolor=col[s], lw=1)
                ax.plot(x[ins], y[ins], "o", color=col[s], ms=5, label=F2.NAME_ALL[s].replace("\n", ", "))
                ax.plot(x[~ins], y[~ins], "o", mfc="none", mec=col[s], ms=5, label=None if ins.any() else F2.NAME_ALL[s].replace("\n", ", "))
            ax.set_xticks(range(len(TEMPL))); ax.set_xticklabels(TEMPL); ax.set_ylabel(f"predicted {F2.UNIT.get(t, t)}")
            ax.spines[["top", "right"]].set_visible(False)
        axs[0][0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_orion.png"), dpi=130); plt.close(fig)
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
