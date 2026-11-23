# Regression for the v2 neutral prototype (2026-11-24): observable sets side by side, and the v2 controls.
# Runs in container C.
#   python slice/regress_v2.py <labels csv> <features csv> <out dir> [--targets logNc,E_eV] [--pysr P,P33,L,L33]
#       [--niter 200] [--procs 16] [--clean csv] [--ctl name=csv ...] [--orion csv]
# Observable sets (all ratios to the 11.2 um band):
#   P     PAHFIT powers, MIRI 5-20 um: r62 r77 r86 r120 r127 r135 r170 (schema v1 of regress.py)
#   P33   P + r33, the 3.3 um band of the NIRSpec segment (local integration) over PAHFIT's 11.2 um model integral
#   L     local-continuum integrals, spline continuum, MIRI: s62 ... s170
#   L33   L + s33
#   LL, LL33      the same with straight-segment continuum (random forest only)
#   P33x, L33x    the 3.3 um ratio times a 5 % (1 sigma, log-normal) cross-calibration error between NIRSpec and MIRI
#   T, T33        noise-free window integrals of the input PAH spectrum (not observable; what the spectrum holds)
# Split, targets and selection rules as regress.py (70/15/15 by the same seeded permutation; everything chosen on
# validation; test scored once). Per set and target: constant, random forest, best one-ratio isotonic fit, PySR
# formulas (complexity <= 7 and <= 12) for the sets named in --pysr.
# Controls:
#   structure     correlation of each predictor's test residual with the population's solo, trio and quartet H shares
#                 and its N and own fractions (v1's size formula: +0.95 with the edge-structure tilt)
#   continuum     test error by continuum type (orion, pl, mbb), and for the forest trained on orion-continuum
#                 populations only and applied to the other two (the v1 out-of-basis test, per measurement method)
#   --ctl name=csv   the same held-out populations measured from another emission variant (harm, tdep900, tdeplin):
#                 error and bias of the predictors trained on the main table, all rows and logNc0 < log10(50)
#   --clean csv   injection-recovery of band/11.2 for PAHFIT and for the local integrals, with and without continuum
#   shuffled labels, deterministic repeat: as regress.py
#   --orion csv   the five Orion Bar templates: kNN domain check per MIRI set (no 3.3 um in the template file)
import argparse, csv, json, os, sys, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regress as R1  # noqa: E402

B = ["62", "77", "86", "120", "127", "135", "170"]
BK = {"62": "6.2", "77": "7.7", "86": "8.6", "112": "11.2", "120": "12.0", "127": "12.7", "135": "13.5", "170": "17", "33": "3.3"}
SETS = {"P": [f"r{b}" for b in B], "L": [f"s{b}" for b in B], "LL": [f"l{b}" for b in B], "T": [f"w{b}" for b in B]}
SETS.update({"P33": SETS["P"] + ["r33"], "L33": SETS["L"] + ["s33"], "LL33": SETS["LL"] + ["l33"], "T33": SETS["T"] + ["w33"],
             "P33x": SETS["P"] + ["r33x"], "L33x": SETS["L"] + ["s33x"]})
STRUCT = ["solo_frac", "trio_frac", "quartet_frac", "N_frac", "own_frac", "cont_level"]
ABS_TARGET = R1.ABS_TARGET
rmse, fl = R1.rmse, R1.fl


def features(rows, pops=None):
    F = R1.features(rows) if "P_PAH_6.2" in rows[0] else {"ok": np.zeros(len(rows), bool), "chi2": np.full(len(rows), np.nan)}
    C = {c: np.array([fl(r.get(c)) for r in rows]) for c in rows[0] if c.startswith(("Llin_", "Lspl_", "Win_", "Wfit_"))}
    with np.errstate(divide="ignore", invalid="ignore"):
        for b in B + ["33"]:
            k = BK[b]
            if f"Lspl_{k}" not in C:
                continue
            num = C[f"Llin_{k}"] if b == "33" else None
            F[f"s{b}"] = (num if b == "33" else C[f"Lspl_{k}"]) / C["Lspl_11.2"]
            F[f"l{b}"] = (num if b == "33" else C[f"Llin_{k}"]) / C["Llin_11.2"]
            if f"Win_{k}" in C:
                F[f"w{b}"] = C[f"Win_{k}"] / C["Win_11.2"]
                F[f"recL{b}"] = F[f"s{b}"] / F[f"w{b}"]; F[f"recLL{b}"] = F[f"l{b}"] / F[f"w{b}"]
        if "Llin_3.3" in C and "Wfit_11.2" in C:
            F["r33"] = C["Llin_3.3"] / C["Wfit_11.2"]
        if "Win_11.2" in C:
            F["recL112"] = C["Lspl_11.2"] / C["Win_11.2"]; F["recLL112"] = C["Llin_11.2"] / C["Win_11.2"]
    if pops is not None and "s33" in F:
        x = np.array([np.exp(np.random.default_rng([7, int(p)]).normal(0, 0.05)) for p in pops])
        F["s33x"] = F["s33"] * x
        if "r33" in F:
            F["r33x"] = F["r33"] * x
    F["okL"] = np.isfinite(C["Lspl_11.2"]) & (C["Lspl_11.2"] > 0) & (C["Llin_11.2"] > 0) if "Lspl_11.2" in C else np.zeros(len(rows), bool)
    return F


def okset(F, name):
    cols = SETS[name]
    if any(c not in F for c in cols):
        return None
    base = F["ok"] if name.startswith("P") else (F["okL"] if name[0] == "L" else np.ones(len(F["okL"]), bool))
    return base & np.all(np.isfinite(R1.mat(F, cols, slice(None))), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("features"); ap.add_argument("out")
    ap.add_argument("--targets", default="logNc,E_eV"); ap.add_argument("--pysr", default="")
    ap.add_argument("--niter", type=int, default=200); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--clean"); ap.add_argument("--ctl", action="append", default=[]); ap.add_argument("--orion")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rf-jobs", type=int, default=16)
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.neighbors import NearestNeighbors
    warnings.filterwarnings("ignore")
    os.makedirs(a.out, exist_ok=True)
    lab = R1.read(a.labels); rows = R1.read(a.features); n = len(rows)
    assert [int(r["pop"]) for r in rows] == list(range(n)) and len(lab) == n
    F = features(rows, np.arange(n))
    L = {c: np.array([fl(r[c]) for r in lab]) for c in lab[0] if c != "cont_type"}
    ctype = np.array([r.get("cont_type", "orion") for r in lab])
    perm = np.random.default_rng(a.seed).permutation(n); n1, n2 = int(0.7 * n), int(0.85 * n)
    part = {"train": perm[:n1], "val": perm[n1:n2], "test": perm[n2:]}
    res = dict(n=n, schema="v2", n_failed_pahfit=int((~F["ok"]).sum()), chi2_red_median=float(np.nanmedian(F["chi2"])), sets={}, controls={})
    print(f"{n} populations; failed PAHFIT fits {res['n_failed_pahfit']}; median reduced chi2 {res['chi2_red_median']:.2f}; "
          f"continuum types { {t: int((ctype == t).sum()) for t in sorted(set(ctype))} }", flush=True)
    allc = sorted({c for v in SETS.values() for c in v if c in F})
    with open(os.path.join(a.out, "features.csv"), "w", newline="") as fo:
        w = csv.writer(fo); sp = {int(i): s for s, ix in part.items() for i in ix}
        w.writerow(["pop", "split", "pahfit_ok", "chi2_red", "cont_type"] + list(L)[1:] + allc)
        for i in range(n):
            w.writerow([i, sp[i], int(F["ok"][i]), f"{F['chi2'][i]:.4g}", ctype[i]] + [f"{L[c][i]:.6g}" for c in list(L)[1:]] + [f"{F[c][i]:.6g}" for c in allc])
    # ---- other tables on held-out populations
    ctl = {}
    for spec in a.ctl + ([f"clean={a.clean}"] if a.clean else []):
        name, path = spec.split("=", 1); r = R1.read(path); pp = np.array([int(x["pop"]) for x in r]); ctl[name] = (features(r, pp), pp)
    rec = {}
    for name, Fo in [("main", F)] + ([("clean", ctl["clean"][0])] if "clean" in ctl else []):
        e = {"n": int(Fo["ok"].sum())}
        for b in ["62", "77", "86", "120", "127", "135"]:
            e[f"pahfit_{b}"] = [float(np.nanpercentile(Fo[f"rec{b}"][Fo["ok"]], q)) for q in (16, 50, 84)]
        for kind in ("L", "LL"):
            for b in B + ["33", "112"]:
                e[f"{kind}_{b}"] = [float(np.nanpercentile(Fo[f"rec{kind}{b}"][Fo["okL"]], q)) for q in (16, 50, 84)]
        e["pahfit_112"] = [float(np.nanpercentile(Fo["rec112"][Fo["ok"]], q)) for q in (16, 50, 84)]
        rec[name] = e
        print(f"recovery of band/11.2 against the input ({name}), median [16, 84 %]; 112 = the 11.2 band itself:")
        for kind, lab_ in (("pahfit", "PAHFIT      "), ("L", "local spline"), ("LL", "local linear")):
            print(f"  {lab_} " + "  ".join(f"{b}: {e[f'{kind}_{b}'][1]:.2f} [{e[f'{kind}_{b}'][0]:.2f},{e[f'{kind}_{b}'][2]:.2f}]"
                                           for b in B + ["33", "112"] if f"{kind}_{b}" in e), flush=True)
    res["injection_recovery"] = rec
    orion = None
    if a.orion:
        r = R1.read(a.orion); orion = (features(r), [x["template"] for x in r], np.array([int(x["mc"]) for x in r]))
    pysr_sets = [x for x in a.pysr.split(",") if x]
    for sname, cols in SETS.items():
        ok = okset(F, sname)
        if ok is None:
            continue
        tr, va, te = [np.array([i for i in part[p] if ok[i]]) for p in ("train", "val", "test")]
        X = lambda Fo, idx: R1.mat(Fo, cols, idx)
        Sres = dict(features=cols, split=dict(train=len(tr), val=len(va), test=len(te)), targets={}); res["sets"][sname] = Sres
        for tgt in a.targets.split(","):
            y = L[tgt]; T = dict(abs_target=ABS_TARGET.get(tgt)); Sres["targets"][tgt] = T
            T["constant"] = dict(val=rmse(y[va], y[tr].mean()), test=rmse(y[te], y[tr].mean()))
            rf = RandomForestRegressor(500, random_state=0, n_jobs=a.rf_jobs).fit(X(F, tr), y[tr])
            preds = {"rf": rf.predict}
            T["rf"] = dict(val=rmse(y[va], rf.predict(X(F, va))), test=rmse(y[te], rf.predict(X(F, te))),
                           importance=dict(zip(cols, np.round(rf.feature_importances_, 3).tolist())))
            one = {}
            for j, c in enumerate(cols):
                m = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(F[c][tr], y[tr])
                one[c] = (m, rmse(y[va], m.predict(F[c][va])))
            b1 = min(one, key=lambda c: one[c][1]); T["one_ratio"] = dict(feature=b1, val=one[b1][1], test=rmse(y[te], one[b1][0].predict(F[b1][te])))
            preds["one_ratio"] = lambda Xm, m=one[b1][0], j=cols.index(b1): m.predict(Xm[:, j])
            if sname in pysr_sets:
                runs, T["pysr_log"] = R1.pysr_runs(X(F, tr), y[tr], cols, a.out, f"{sname}_{tgt}", a.niter, a.procs)
                T["deterministic_reproduces"] = ([r["equation"] for r in runs["det"]] == [r["equation"] for r in runs["det_repeat"]]
                                                 if "det" in runs and "det_repeat" in runs else None)
                cand = []
                for name, fr in runs.items():
                    for r in fr:
                        p = R1.evalf(r["f"], X(F, va)); r["val"] = rmse(y[va], p) if np.all(np.isfinite(p)) else float("inf"); r["run"] = name
                        if name not in ("shuffled", "det_repeat"):
                            cand.append(r)
                T["shuffled_best_val"] = min(r["val"] for r in runs["shuffled"]) if "shuffled" in runs else None
                for label, cmax in (("compact", 7), ("full", 12)):
                    cc = [r for r in cand if r["complexity"] <= cmax]
                    if not cc:
                        continue
                    c = min(cc, key=lambda r: r["val"]); pt = R1.evalf(c["f"], X(F, te))
                    T[f"formula_{label}"] = dict(equation=c["equation"], complexity=c["complexity"], run=c["run"], val=c["val"],
                                                 test=rmse(y[te], pt), variables=R1.used_vars(c["equation"], cols))
                    preds[f"formula_{label}"] = lambda Xm, f=c["f"]: R1.evalf(f, Xm)
                    T[f"formula_{label}"]["vs_rf"] = T[f"formula_{label}"]["test"] / T["rf"]["test"]
                    T[f"formula_{label}"]["vs_one_ratio"] = T[f"formula_{label}"]["test"] / T["one_ratio"]["test"]
            # ---- diagnostics of every predictor on the test rows
            for pn, pf in preds.items():
                p = pf(X(F, te)); r = p - y[te]; g = np.isfinite(r); d = T[pn]
                d["resid_corr"] = {c: float(np.corrcoef(r[g], L[c][te][g])[0, 1]) for c in STRUCT if c in L and L[c][te].std() > 0}
                d["test_by_cont"] = {t: rmse(y[te][ctype[te] == t], p[ctype[te] == t]) for t in sorted(set(ctype))}
                d["bias_by_cont"] = {t: float(np.mean(r[ctype[te] == t])) for t in sorted(set(ctype))}
            # ---- forest trained on the orion continuum only
            if len(set(ctype)) > 1:
                tro = tr[ctype[tr] == "orion"]; m = RandomForestRegressor(500, random_state=0, n_jobs=a.rf_jobs).fit(X(F, tro), y[tro])
                p = m.predict(X(F, te))
                T["rf_orion_only"] = dict(n_train=len(tro), test_by_cont={t: rmse(y[te][ctype[te] == t], p[ctype[te] == t]) for t in sorted(set(ctype))},
                                          bias_by_cont={t: float(np.mean((p - y[te])[ctype[te] == t])) for t in sorted(set(ctype))})
            # ---- other emission variants, same held-out populations
            for cname, (Fo, pp) in ctl.items():
                oko = okset(Fo, sname)
                if cname == "clean" or oko is None:
                    continue
                pos = {int(q): k for k, q in enumerate(pp) if oko[k]}
                ev = np.array([i for i in np.concatenate([va, te]) if i in pos]); ko = np.array([pos[i] for i in ev])
                small = L["logNc0"][ev] < np.log10(50.0); d = dict(n=len(ev), n_small=int(small.sum()))
                for pn, pf in preds.items():
                    pm, po = pf(X(F, ev)), pf(X(Fo, ko))
                    d[pn] = dict(main=rmse(y[ev], pm), other=rmse(y[ev], po), bias=float(np.mean(po - y[ev])), shift=float(np.mean(po - pm)),
                                 shift_small=float(np.mean((po - pm)[small])), shift_large=float(np.mean((po - pm)[~small])))
                T[f"control_{cname}"] = d
            # ---- Orion templates: domain only
            if orion and okset(orion[0], sname) is not None:
                Fo, tn, mc = orion; Xo = X(Fo, slice(None)); mu, sd = X(F, tr).mean(0), X(F, tr).std(0); Z = (X(F, tr) - mu) / sd
                nn = NearestNeighbors(n_neighbors=6).fit(Z); thr = float(np.percentile(nn.kneighbors(Z)[0][:, 5], 99))
                do = nn.kneighbors(np.nan_to_num((Xo - mu) / sd, nan=99.0), n_neighbors=5)[0][:, 4]
                T["orion"] = {t: dict(knn_dist=float(do[k]), knn_threshold=thr, extrapolation=bool(do[k] > thr),
                                      features={c: float(Fo[c][k]) for c in cols}, rf=float(rf.predict(np.nan_to_num(Xo[k:k + 1]))[0]))
                              for k, t in enumerate(tn) if mc[k] == 0}
            # ---- held-out predictions
            with open(os.path.join(a.out, f"pred_{sname}_{tgt}.csv"), "w", newline="") as fo:
                w = csv.writer(fo); w.writerow(["pop", "split", "cont_type", "y"] + list(preds))
                for sp_, ix in (("val", va), ("test", te)):
                    P = [pf(X(F, ix)) for pf in preds.values()]
                    w.writerows([[int(i), sp_, ctype[i], f"{y[i]:.5g}"] + [f"{p[k]:.5g}" for p in P] for k, i in enumerate(ix)])
            s = f"[{sname:5s} {tgt:6s}] const {T['constant']['test']:.4f} | one-ratio {b1} {T['one_ratio']['test']:.4f} | RF {T['rf']['test']:.4f}"
            for label in ("compact", "full"):
                if f"formula_{label}" in T:
                    d = T[f"formula_{label}"]; s += f" | c{d['complexity']} {d['test']:.4f}"
            print(s + "   (test RMSE)", flush=True)
            for label in ("compact", "full"):
                if f"formula_{label}" in T:
                    d = T[f"formula_{label}"]
                    print(f"      {label}: {d['equation']}   residual corr { {k: round(v, 2) for k, v in d['resid_corr'].items()} }", flush=True)
            if "pysr_log" in T:
                print(f"      det repeat {T['deterministic_reproduces']}, shuffled val {T['shuffled_best_val']} (constant {T['constant']['val']:.4f}), "
                      f"crashes { {k: v['crashes'] for k, v in T['pysr_log'].items()} }", flush=True)
            json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    print(f"\nwrote {a.out}/results.json, features.csv, pred_*.csv")


if __name__ == "__main__":
    main()
