# Tables of reports/nc_sr_v5/README.md from the result files of regress_v2.py (version 5 against version 4).
#   python slice/report_v5.py <v5 results_base.json> <v5 results_s112.json> <v4 results_base.json> [<v4 results_pfit.json>]
import json, sys

TG = [("f_cation", "f", 3), ("logNc", "log Nc (dex)", 3), ("E_eV", "E (eV)", 2)]
SETS = [("Hn", "Lpch, MIRI"), ("H33n", "Lpch, NIRSpec + MIRI"), ("Pn", "PAHFIT, MIRI"), ("P33n", "PAHFIT, NIRSpec + MIRI")]
TPL = ["HII", "APDR", "DF1", "DF2", "DF3"]


def load(paths):
    S = {}
    for p in paths:
        for k, v in json.load(open(p))["sets"].items():
            if k not in S or any("formula_full" in t for t in v["targets"].values()):
                S[k] = v
    return S


def g(S, s, t, *keys):
    d = S.get(s, {}).get("targets", {}).get(t)
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def f(x, n=3):
    return "–" if x is None else f"{x:.{n}f}"


def main():
    v5 = load([sys.argv[1]]); s112 = load([sys.argv[2]]); v4 = load(sys.argv[3:])
    print("### Test error (held-out 300 populations), v4 | v5\n")
    print("| Observables | Predictor | " + " | ".join(f"{n}: v4 | v5" for _, n, _ in TG) + " |")
    print("| --- | --- | " + " | ".join("---: | ---:" for _ in TG) + " |")
    for s, sn in SETS:
        for pk, pn in (("rf", "forest"), ("one_ratio", "best single ratio"), ("formula_compact", "formula, complexity ≤ 7"), ("formula_full", "formula, complexity ≤ 12")):
            print(f"| {sn} | {pn} | " + " | ".join(f"{f(g(v4, s, t, pk, 'test'), d)} | {f(g(v5, s, t, pk, 'test'), d)}" for t, _, d in TG) + " |")
    print("| – | constant (no information) | " + " | ".join(f"{f(g(v4, 'Hn', t, 'constant', 'test'), d)} | {f(g(v5, 'Hn', t, 'constant', 'test'), d)}" for t, _, d in TG) + " |")
    print("\n### Formulas\n")
    print("| Observables | Target | Version | Formula (complexity) | Test | Forest | Reproduced; shuffled-label val |")
    print("| --- | --- | --- | --- | ---: | ---: | --- |")
    for s, sn in SETS:
        for t, tn, d in TG:
            for vn, V in (("v4", v4), ("v5", v5)):
                for pk in ("formula_compact", "formula_full"):
                    e = g(V, s, t, pk)
                    if e:
                        print(f"| {sn} | {tn} | {vn} | `{e['equation']}` ({e['complexity']}) | {f(e['test'], d)} | {f(g(V, s, t, 'rf', 'test'), d)} | "
                              f"{'yes' if g(V, s, t, 'deterministic_reproduces') else 'no'}; {f(g(V, s, t, 'shuffled_best_val'), d)} |")
    print("\n### Other observable sets, forest test error (v4 | v5)\n")
    print("| Set | " + " | ".join(f"{n}: v4 | v5" for _, n, _ in TG) + " |"); print("| --- | " + " | ".join("---: | ---:" for _ in TG) + " |")
    for s in v5:
        print(f"| {s} | " + " | ".join(f"{f(g(v4, s, t, 'rf', 'test'), d)} | {f(g(v5, s, t, 'rf', 'test'), d)}" for t, _, d in TG) + " |")
    print("\n### Sensitivity: cation 11.2 um strength slope -4.7 (L = 16 ps) against -0.4 % per 100 K (5.1 ps)\n")
    print("Held-out 600 populations. 'applied' = the model trained on the 16 ps spectra applied to the 5.1 ps spectra; 'retrained' = trained and tested on the 5.1 ps spectra.\n")
    print("| Observables | Predictor | " + " | ".join(f"{n}: 16 ps | applied | shift | retrained (test)" for _, n, _ in TG) + " |")
    print("| --- | --- | " + " | ".join("---: | ---: | ---: | ---:" for _ in TG) + " |")
    for s, sn in SETS:
        for pk, pn in (("rf", "forest"), ("formula_full", "formula ≤ 12")):
            cells = []
            for t, _, d in TG:
                c = g(v5, s, t, "control_s112", pk)
                cells.append(f"{f(c and c['main'], d)} | {f(c and c['other'], d)} | {'–' if not c else format(c['shift'], f'+.{d}f')} | {f(g(s112, s, t, pk, 'test'), d)}")
            print(f"| {sn} | {pn} | " + " | ".join(cells) + " |")
    print("\n### Monte Carlo error bars (S/N 50 at 11.2 um), rms scatter of the prediction over 10 noise draws\n")
    print("| Observables | Predictor | " + " | ".join(n for _, n, _ in TG) + " |"); print("| --- | --- | " + " | ".join("---:" for _ in TG) + " |")
    for s, sn in SETS:
        for pk, pn in (("rf", "forest"), ("formula_full", "formula ≤ 12")):
            print(f"| {sn} | {pn} | " + " | ".join(f"{f(g(v5, s, t, 'mc_s50', pk, 'sd_rms'), d)}" for t, _, d in TG) + " |")
    print("\n### Orion Bar templates (reported, never a gate): forest | formula ≤ 12, v4 and v5\n")
    for t, tn, d in TG:
        print(f"\n**{tn}**\n\n| Observables | Version | " + " | ".join(TPL) + " | outside the library |"); print("| --- | --- | " + " | ".join("---:" for _ in TPL) + " | --- |")
        for s, sn in SETS:
            for vn, V in (("v4", v4), ("v5", v5)):
                O = g(V, s, t, "orion")
                if O:
                    out = sorted({c for x in TPL for c in O[x]["outside_train_1_99"]} | ({"kNN"} if any(O[x]["extrapolation"] for x in TPL) else set()))
                    print(f"| {sn} | {vn} | " + " | ".join(f"{f(O[x]['rf'], 2)} ± {f(O[x].get('rf_mc_sd'), 2)} \\| {f(O[x].get('formula_full'), 2)}" for x in TPL) + f" | {', '.join(out) or 'none'} |")
    print("\n### Orion refits (slice/orion_cont.py): cation fraction, forest of v5\n")
    print("| Fit | Set | " + " | ".join(TPL) + " |"); print("| --- | --- | " + " | ".join("---:" for _ in TPL) + " |")
    for pk in ("", "@classic", "@hot", "@plat", "@hotplat", "@sub", "@anc5"):
        for s in ("Pn", "P33n", "Hn", "H33n", "Hc", "Hc33"):
            O = g(v5, s, "f_cation", "orion")
            if O and all(x + pk in O for x in TPL):
                print(f"| {pk[1:] or 'measure.py'} | {s} | " + " | ".join(f(O[x + pk]["rf"], 2) for x in TPL) + " |")
    print("\n### 17 um complex above the curved continuum (c170 = Lcrv_17 / Lpch_11.2)\n")
    O = g(v5, "Hc", "f_cation", "orion"); r = v5.get("Hc", {}).get("targets", {}).get("f_cation", {}).get("train_1_99", {}).get("c170")
    if O:
        print("library 1-99 %:", r, "; templates:", {x: round(O[x]["features"]["c170"], 3) for x in TPL})
        for s in ("Hn", "Hc", "H33n", "Hc33"):
            print(s, {t: (round(g(v5, s, t, "rf", "test"), 4), "importance c170", (g(v5, s, t, "rf", "importance") or {}).get("c170")) for t, _, _ in TG},
                  "Orion f:", {x: round(g(v5, s, "f_cation", "orion")[x]["rf"], 2) for x in TPL})


if __name__ == "__main__":
    main()
