# Figures for the v2 report. Runs in container C.
#   python slice/figures_v2.py <regress_v2 out dir> <fig dir>
# fig_sets.png   test RMSE per observable set and target: constant, best single ratio, random forest, formulas
# fig_pred.png   held-out test rows, predicted against true, for the sets with formulas
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAME = {"P": "PAHFIT\nMIRI", "P33": "PAHFIT\n+ 3.3", "L": "local\nMIRI", "L33": "local\n+ 3.3", "T": "noise-free\nMIRI", "T33": "noise-free\n+ 3.3"}
NAME_ALL = dict(NAME); NAME_ALL.update({"Ln": "local MIRI\nno 17", "L33n": "local + 3.3\nno 17", "Pn": "PAHFIT MIRI\nno 17", "P33n": "PAHFIT + 3.3\nno 17",
                                      "Hn": "local (PCHIP) MIRI\nno 17", "H33n": "local (PCHIP) + 3.3\nno 17", "LLn": "local (straight) MIRI\nno 17",
                                      "LL33n": "local (straight) + 3.3\nno 17", "H": "local (PCHIP)\nMIRI", "H33": "local (PCHIP)\n+ 3.3"})
UNIT = {"logNc": "log$_{10}$ N$_C$ (dex)", "E_eV": "photon energy (eV)", "f_cation": "cation fraction"}
KIND = [("one_ratio", "best single ratio", "#9aa0a6"), ("rf", "random forest", "#1f6feb"), ("formula_compact", "formula, complexity ≤ 7", "#d1495b"),
        ("formula_full", "formula, complexity ≤ 12", "#edae49")]


def main(reg, out):
    os.makedirs(out, exist_ok=True); res = json.load(open(os.path.join(reg, "results.json")))
    sets = [s for s in NAME if s in res["sets"]]; tg = list(res["sets"][sets[0]]["targets"])
    fig, axs = plt.subplots(1, len(tg), figsize=(6.2 * len(tg), 4.2), squeeze=False)
    for ax, t in zip(axs[0], tg):
        w = 0.2
        for k, (key, lab, col) in enumerate(KIND):
            xs = [i + (k - 1.5) * w for i, s in enumerate(sets) if key in res["sets"][s]["targets"][t]]
            ys = [res["sets"][s]["targets"][t][key]["test"] for s in sets if key in res["sets"][s]["targets"][t]]
            ax.bar(xs, ys, w * 0.92, color=col, label=lab)
        c = res["sets"][sets[0]]["targets"][t]["constant"]["test"]; ax.axhline(c, color="0.3", lw=0.8, ls="--"); ax.text(len(sets) - 0.5, c, " constant", va="bottom", ha="right", fontsize=8)
        if res["sets"][sets[0]]["targets"][t].get("abs_target"):
            a = res["sets"][sets[0]]["targets"][t]["abs_target"]; ax.axhline(a, color="0.3", lw=0.8, ls=":"); ax.text(len(sets) - 0.5, a, " target", va="bottom", ha="right", fontsize=8)
        ax.set_xticks(range(len(sets))); ax.set_xticklabels([NAME[s] for s in sets], fontsize=8)
        ax.set_ylabel(f"test RMSE, {UNIT.get(t, t)}"); ax.spines[["top", "right"]].set_visible(False)
    axs[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_sets.png"), dpi=130); plt.close(fig)
    fs = [s for s in sets if any(k.startswith("formula") for k in res["sets"][s]["targets"][tg[0]])]
    if fs:
        fig, axs = plt.subplots(len(tg), len(fs), figsize=(3.6 * len(fs), 3.5 * len(tg)), squeeze=False)
        for i, t in enumerate(tg):
            for j, s in enumerate(fs):
                ax = axs[i][j]; rows = [r for r in csv.DictReader(open(os.path.join(reg, f"pred_{s}_{t}.csv"))) if r["split"] == "test"]
                y = np.array([float(r["y"]) for r in rows])
                for key, lab, col in KIND[1:3]:
                    if key in rows[0]:
                        ax.plot(y, [float(r[key]) for r in rows], ".", color=col, ms=3, alpha=0.6, label=f"{lab} ({res['sets'][s]['targets'][t][key]['test']:.3g})")
                ax.plot([y.min(), y.max()], [y.min(), y.max()], "k-", lw=0.8); pad = 0.25 * (y.max() - y.min()); ax.set_ylim(y.min() - pad, y.max() + pad)
                ax.set_title(NAME[s].replace("\n", ", "), fontsize=9); ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)
                ax.set_xlabel(f"true {UNIT.get(t, t)}", fontsize=8); ax.set_ylabel("predicted", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_pred.png"), dpi=130); plt.close(fig)
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
