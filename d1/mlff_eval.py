# Held-out errors of one or more MACE models (two heads q0/q1) on a curated dataset (d1/mlff_dataset.py), side by side;
# docs/plan/charged_mlff_plan.md, Phase 2/3. Container B.
#   python mlff_eval.py <dataset dir> <out.json> --models NAME=path[,NAME=path...] [--splits validation,final_test,
#        transfer_test] [--files extra.extxyz,...] [--device cuda]
# Per model, split (or extra file) and charge: force RMSE (meV/A, over all components), energy RMSE (meV/atom) and the
# energy RMSE after removing each pair's mean energy error (meV/atom; the cation head's IP-like per-molecule offset,
# which a sum of atomic terms cannot hold, is then excluded), plus force RMSE per config_type. --files evaluates any
# labelled extxyz (e.g. an AL batch: how wrong the old model was where the committee disagreed); its charge comes
# from each frame. Prints Markdown tables; the JSON keeps the per-frame errors.
# Gate G2 (docs/plan/charged_mlff_proposal.md 5.7) is a force MAE <= 60 meV/A on held-out species per charge state and
# sampling temperature, so the force MAE (over all components) is reported too: per set, and per charge and
# config_type pooled over the held-out splits (not the --files sets), each marked against G2_F_MAE. Per frame the JSON
# keeps sF (sum of squared force errors) and aF (sum of absolute ones), in eV/A, so any subset can be recomputed.
import argparse, collections, json, os
import numpy as np

G2_F_MAE = 60.0   # meV/A


def frames(path):
    from ase.io import read
    return read(path, ":") if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("out"); ap.add_argument("--models", required=True)
    ap.add_argument("--splits", default="validation,final_test,transfer_test"); ap.add_argument("--files", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from mace.calculators import MACECalculator
    models = dict(x.split("=", 1) for x in a.models.split(","))
    sets = {f"{sp}_q{q}": frames(f"{a.ds}/{sp}_q{q}.extxyz") for sp in a.splits.split(",") for q in (0, 1)}
    for f in filter(None, a.files.split(",")):
        fr = frames(f); name = os.path.basename(f).rsplit(".", 1)[0]
        for q in (0, 1):
            sets[f"{name}_q{q}"] = [x for x in fr if int(x.info["charge"]) == q]
    res = {}
    for mname, path in models.items():
        calc = {q: MACECalculator(model_paths=path, device=a.device, default_dtype="float32", head=f"q{q}") for q in (0, 1)}
        for c in calc.values():
            assert c.available_heads == ["q0", "q1"], c.available_heads
        for sname, fr in sets.items():
            rows = []
            for x in fr:
                q = int(x.info["charge"]); Ed, Fd = x.info["dft_energy"], x.arrays["dft_forces"].copy()
                y = x.copy(); y.calc = calc[q]
                rows.append(dict(pair=x.info["source"].rsplit("_q", 1)[0], ct=x.info.get("config_type", "?"), n=len(x),
                                 dE=float(y.get_potential_energy() - Ed), sF=float(((y.get_forces() - Fd) ** 2).sum()),
                                 aF=float(np.abs(y.get_forces() - Fd).sum())))
            res[(mname, sname)] = rows
            print(f"{mname} {sname}: {len(rows)} frames", flush=True)
    summ = {}
    for (m, s), rows in res.items():
        if not rows:
            continue
        nF = sum(3 * r["n"] for r in rows)
        mean_by_pair = collections.defaultdict(list)
        for r in rows:
            mean_by_pair[r["pair"]].append(r["dE"])
        off = {p: np.mean(v) for p, v in mean_by_pair.items()}
        ct = collections.defaultdict(lambda: [0.0, 0])
        for r in rows:
            ct[r["ct"]][0] += r["sF"]; ct[r["ct"]][1] += 3 * r["n"]
        summ[f"{m}|{s}"] = dict(
            frames=len(rows), F=1e3 * np.sqrt(sum(r["sF"] for r in rows) / nF),
            F_mae=1e3 * sum(r["aF"] for r in rows) / nF,
            E=1e3 * np.sqrt(np.mean([(r["dE"] / r["n"]) ** 2 for r in rows])),
            E_rel=1e3 * np.sqrt(np.mean([((r["dE"] - off[r["pair"]]) / r["n"]) ** 2 for r in rows])),
            F_by_ct={k: 1e3 * np.sqrt(v[0] / v[1]) for k, v in sorted(ct.items())})
    ms = list(models); ss = [s for s in sets if any(f"{m}|{s}" in summ for m in ms)]
    for key, unit in (("F", "force RMSE, meV/A"), ("F_mae", f"force MAE, meV/A (G2: <= {G2_F_MAE:.0f})"),
                      ("E", "energy RMSE, meV/atom"),
                      ("E_rel", "energy RMSE after per-pair offset, meV/atom")):
        print(f"\n### {unit}\n\n| set | frames | " + " | ".join(ms) + " |\n|---|---|" + "---|" * len(ms))
        for s in ss:
            n = next(summ[f"{m}|{s}"]["frames"] for m in ms if f"{m}|{s}" in summ)
            print(f"| {s} | {n} | " + " | ".join(f"{summ[f'{m}|{s}'][key]:.1f}" if f"{m}|{s}" in summ else "-" for m in ms) + " |")
    held = [f"{sp}_q{q}" for sp in a.splits.split(",") for q in (0, 1)]
    g2 = {}
    for q in (0, 1):
        cts = sorted({r["ct"] for (m, s), rows in res.items() if s in held and s.endswith(f"_q{q}") for r in rows})
        print(f"\n### force MAE per config_type, q{q}, held-out splits pooled, meV/A (* = above G2 {G2_F_MAE:.0f})\n\n"
              "| config_type | frames | " + " | ".join(ms) + " |\n|---|---|" + "---|" * len(ms))
        for ct in cts:
            cells, nfr = [], 0
            for m in ms:
                rows = [r for s in held if s.endswith(f"_q{q}") for r in res.get((m, s), []) if r["ct"] == ct]
                if not rows:
                    cells.append("-"); continue
                v = 1e3 * sum(r["aF"] for r in rows) / sum(3 * r["n"] for r in rows); nfr = len(rows)
                g2[f"{m}|q{q}|{ct}"] = v; cells.append(f"{v:.1f}" + ("*" if v > G2_F_MAE else ""))
            print(f"| {ct} | {nfr} | " + " | ".join(cells) + " |")
    json.dump(dict(models=models, summary=summ, g2_F_mae_by_ct=g2, frames={f"{m}|{s}": r for (m, s), r in res.items()}),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
