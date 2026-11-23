# Feature table, random-forest reference, one-ratio baselines, PySR and the controls (prototype stages 6-9;
# plan 5.7-5.10). Runs in container C.
#   python slice/regress.py <labels csv> <features csv> <out dir> [--targets logNc,E_eV] [--pysr] [--niter 200]
#       [--procs 16] [--clean csv] [--oob csv] [--other csv --other-name harm] [--orion csv] [--f-target]
# Frozen feature schema v1 (PAHFIT classic.yaml powers; complexes summed by group):
#   bands    6.2 = PAH_6.2, 7.7 = PAH_7.7a+b+c, 8.6 = PAH_8.6, 11.2 = PAH_11.3a+b, 12.0 = PAH_12, 12.7 = PAH_12.6a+b,
#            13.5 = PAH_13.48, 17 = PAH_17a+b+c+d
#   primary  r62, r77, r86, r120, r127, r135, r170: band / 11.2 (the 11.2 um band is strong in every population
#            of the neutral library: share of 5-20 um emission >= 0.15, populations.py)
#   second   t62 ... t170, t112: band / total, total = all dust features from 5.27 to 17.87 um (PAH_33.1 left out)
#   no_oop   r62, r77, r86, r170 only (random forest): how much of the answer comes from the edge-structure bands
#   truth    w62 ... w135: the same ratios from window integrals of the noise-free input spectrum (not observable;
#            a reference for what the decomposition loses)
# A fit counts as failed when PAHFIT raised, stopped on maxfev, or returned a non-positive or non-finite 11.2 um
# power; failed rows are counted and left out of training and scoring, never silently.
# Split: 70/15/15 by a seeded permutation of the population index. Everything is selected on validation; the test
# set is scored once at the end.
# Targets fixed before the run: RMSE(logNc) <= 0.05 dex, RMSE(E) <= 1.5 eV, RMSE(f) <= 0.1.
import argparse, csv, json, os, sys, time, warnings
import numpy as np

BANDS = {"62": ["PAH_6.2"], "77": ["PAH_7.7a", "PAH_7.7b", "PAH_7.7c"], "86": ["PAH_8.6"], "112": ["PAH_11.3a", "PAH_11.3b"],
         "120": ["PAH_12"], "127": ["PAH_12.6a", "PAH_12.6b"], "135": ["PAH_13.48"], "170": ["PAH_17a", "PAH_17b", "PAH_17c", "PAH_17d"]}
TOTAL_EXCLUDE = ("P_PAH_33.1",)
PRIMARY = ["r62", "r77", "r86", "r120", "r127", "r135", "r170"]
SECOND = ["t62", "t77", "t86", "t112", "t120", "t127", "t135", "t170"]
TRUTH = ["w62", "w77", "w86", "w120", "w127", "w135"]
NO_OOP = ["r62", "r77", "r86", "r170"]        # without the 12.0/12.7/13.5 um ratios, which measure edge structure (duo/trio/quartet H)
WKEY = {"62": "6.2", "77": "7.7", "86": "8.6", "112": "11.2", "120": "12.0", "127": "12.7", "135": "13.5"}
ABS_TARGET = {"logNc": 0.05, "E_eV": 1.5, "f_cation": 0.1}


def read(path):
    return list(csv.DictReader(open(path)))


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def features(rows):
    """dict of arrays: ok flag, primary/second/truth features, raw band powers."""
    P = {c: np.array([fl(r.get(c)) for r in rows]) for c in rows[0] if c.startswith(("P_", "Win_", "Wfit_"))}
    B = {b: sum(P[f"P_{n}"] for n in names) for b, names in BANDS.items()}
    tot = sum(v for c, v in P.items() if c.startswith("P_") and c not in TOTAL_EXCLUDE)
    conv = np.array([fl(r.get("converged")) == 1 for r in rows])
    ok = conv & np.isfinite(B["112"]) & (B["112"] > 0) & np.isfinite(tot)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = {f"r{b}": B[b] / B["112"] for b in BANDS if b != "112"}
        F.update({f"t{b}": B[b] / tot for b in BANDS})
        if "Win_11.2" in P:
            F.update({f"w{b}": P[f"Win_{WKEY[b]}"] / P["Win_11.2"] for b in WKEY if b != "112"})
            F.update({f"rec{b}": (P[f"Wfit_{WKEY[b]}"] / P["Wfit_11.2"]) / (P[f"Win_{WKEY[b]}"] / P["Win_11.2"]) for b in WKEY if b != "112"})
            F["rec112"] = P["Wfit_11.2"] / P["Win_11.2"]
    F["ok"] = ok; F["chi2"] = np.array([fl(r.get("chi2_red")) for r in rows])
    return F


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mat(F, cols, idx):
    return np.stack([F[c][idx] for c in cols], 1)


def _pysr_child(q, Xtr, y, names, common, kw, tag):
    os.environ["PAH_PYSR_TAG"] = tag                    # inherited by Julia's worker processes, so they can be found
    os.environ["JULIA_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"   # before Julia starts (2026-11-24)
    os.environ["PYTHON_JULIACALL_THREADS"] = "1"; os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    from pysr import PySRRegressor
    m = PySRRegressor(**common, **kw); m.fit(Xtr, y, variable_names=names)                # unweighted (plan 5.8)
    eq = m.equations_
    q.put([(int(c), float(l), str(e)) for c, l, e in zip(eq["complexity"], eq["loss"], eq["equation"])])


def kill_tagged(tag):
    """Kill every process started with PAH_PYSR_TAG=tag in its environment: Julia workers outlive a crashed fit,
    keep the parent's pipes open and make it hang at exit."""
    import signal
    for pid in os.listdir("/proc"):
        if pid.isdigit() and int(pid) != os.getpid():
            try:
                if f"PAH_PYSR_TAG={tag}".encode() in open(f"/proc/{pid}/environ", "rb").read():
                    os.kill(int(pid), signal.SIGKILL)
            except (OSError, PermissionError):
                pass


def compile_eq(eq, names):
    """PySR's equation string (+ - * /, log, sqrt, square, variable names) as a function of the feature matrix."""
    code = compile(eq, "<equation>", "eval"); env = {"log": np.log, "sqrt": np.sqrt, "square": np.square, "__builtins__": {}}
    return lambda X: eval(code, dict(env, **{n: X[:, j] for j, n in enumerate(names)}))


def pysr_runs(Xtr, ytr, names, outdir, tag, niter, procs, seeds=(1, 2, 3), tries=4):
    """Every fit runs in its own process: Julia can segfault inside SymbolicRegression (seen 2026-11-23, in
    ConstantOptimization and in the multiprocessing serializer), which would otherwise kill the whole stage. A
    crashed fit is retried with the same settings (the crashes are not tied to the seed); the count is kept in `log`."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    common = dict(binary_operators=["+", "-", "*", "/"], unary_operators=["log", "sqrt", "square"], maxsize=12, niterations=niter,
                  populations=32, population_size=50, model_selection="best", constraints={"/": (-1, 4)},
                  nested_constraints={"log": {"log": 0, "sqrt": 0}, "sqrt": {"sqrt": 0}}, progress=False, verbosity=0,
                  turbo=False, bumper=False,              # process-only parallelism (`parallelism` below), no LoopVectorization/Bumper
                  output_directory=os.path.join(outdir, "pysr_runs"))
    runs, log = {}, {}

    def one(name, y, random_state, **kw):
        for k in range(tries):
            t = time.time(); q = ctx.Queue(); out = None
            tg = f"{os.getpid()}-{tag}-{name}-{k}"
            pr = ctx.Process(target=_pysr_child, args=(q, Xtr, y, names, common, dict(kw, random_state=random_state), tg))
            pr.start()
            while out is None and time.time() - t < 3600:
                try:
                    out = q.get(timeout=5)
                except Exception:                                                     # queue empty: still running, or dead
                    if not pr.is_alive():
                        try:
                            out = q.get(timeout=5)
                        except Exception:
                            pass
                        break
            pr.join(60)
            if pr.is_alive():
                pr.kill(); pr.join(10)
            kill_tagged(tg)
            if out is not None:
                break
            print(f"    PySR {tag} {name}: crashed (exit code {pr.exitcode}), try {k + 1} of {tries}", flush=True)
        log[name] = dict(random_state=random_state, crashes=k if out is not None else tries, ok=out is not None)
        if out is None:
            return
        fr = [dict(complexity=c, loss=l, equation=e, f=compile_eq(e, names)) for c, l, e in out]
        with open(os.path.join(outdir, f"front_{tag}_{name}.csv"), "w", newline="") as fo:
            w = csv.writer(fo); w.writerow(["complexity", "train_loss", "equation"])
            w.writerows([[r["complexity"], f"{r['loss']:.6g}", r["equation"]] for r in fr])
        print(f"    PySR {tag} {name}: {time.time() - t:.0f} s, {len(fr)} front members", flush=True)
        runs[name] = fr
    one("det", ytr, 0, deterministic=True, parallelism="serial")
    one("det_repeat", ytr, 0, deterministic=True, parallelism="serial")
    for s in seeds:
        one(f"seed{s}", ytr, s, deterministic=False, parallelism="multiprocessing", procs=procs)
    one("shuffled", np.random.default_rng(99).permutation(ytr), 4, deterministic=False, parallelism="multiprocessing", procs=procs)
    return runs, log


def evalf(f, X):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            y = np.asarray(f(X), float)
        except Exception:
            return np.full(len(X), np.nan)
    return np.broadcast_to(y, (len(X),)).astype(float)


def used_vars(eq, names):
    import re
    return sorted({n for n in names if re.search(rf"(?<![A-Za-z0-9_]){n}(?![0-9])", eq)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("features"); ap.add_argument("out")
    ap.add_argument("--targets", default="logNc,E_eV"); ap.add_argument("--pysr", action="store_true")
    ap.add_argument("--niter", type=int, default=200); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--clean"); ap.add_argument("--oob"); ap.add_argument("--other"); ap.add_argument("--other-name", default="harm")
    ap.add_argument("--orion"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rf-jobs", type=int, default=16)
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.neighbors import NearestNeighbors
    os.makedirs(a.out, exist_ok=True)
    lab = read(a.labels); rows = read(a.features)
    assert [int(r["pop"]) for r in rows] == list(range(len(rows))) and len(rows) == len(lab)
    F = features(rows); n = len(rows)
    L = {c: np.array([fl(r[c]) for r in lab]) for c in lab[0]}
    perm = np.random.default_rng(a.seed).permutation(n); n1, n2 = int(0.7 * n), int(0.85 * n)
    ok = F["ok"] & np.all(np.isfinite(mat(F, PRIMARY, slice(None))), 1)
    tr, va, te = [np.array([i for i in part if ok[i]]) for part in (perm[:n1], perm[n1:n2], perm[n2:])]
    res = dict(n=n, n_failed=int((~ok).sum()), split=dict(train=len(tr), val=len(va), test=len(te)), schema="v1",
               chi2_red_median=float(np.nanmedian(F["chi2"])), targets={})
    print(f"{n} populations; failed fits {res['n_failed']}; train/val/test {len(tr)}/{len(va)}/{len(te)}; "
          f"median reduced chi2 {res['chi2_red_median']:.2f}", flush=True)
    feat_tab = os.path.join(a.out, "features.csv")
    with open(feat_tab, "w", newline="") as fo:
        cols = PRIMARY + SECOND + [c for c in TRUTH if c in F]
        w = csv.writer(fo); w.writerow(["pop", "split", "fit_ok", "chi2_red"] + [c for c in L if c != "pop"] + cols)
        part = {int(i): s for s, ix in (("train", perm[:n1]), ("val", perm[n1:n2]), ("test", perm[n2:])) for i in ix}
        for i in range(n):
            w.writerow([i, part[i], int(ok[i]), f"{F['chi2'][i]:.4g}"] + [f"{L[c][i]:.6g}" for c in L if c != "pop"] + [f"{F[c][i]:.6g}" for c in cols])
    # other feature tables (same populations)
    other = {}
    for name, path in (("clean", a.clean), ("oob", a.oob), (a.other_name, a.other)):
        if path:
            r = read(path); other[name] = (features(r), np.array([int(x["pop"]) for x in r]))
    if "clean" in other or True:
        rec = {}
        for name, (Fo, pops) in [("main", (F, np.arange(n)))] + [(k, v) for k, v in other.items() if k in ("clean", "oob")]:
            g = Fo["ok"]
            rec[name] = {b: [float(np.nanpercentile(Fo[f"rec{b}"][g], q)) for q in (16, 50, 84)] for b in WKEY}
            rec[name]["n"] = int(g.sum())
        res["injection_recovery"] = rec
        print("recovery of band/11.2 from PAHFIT's PAH model against the input, same windows (16/50/84 %); 112 = the 11.2 band itself:")
        for name, v in rec.items():
            print(f"  {name:6s} n={v['n']:4d} " + "  ".join(f"{b}: {v[b][1]:.2f} [{v[b][0]:.2f},{v[b][2]:.2f}]" for b in WKEY), flush=True)
    orion = None
    if a.orion:
        r = read(a.orion); orion = (features(r), [x["template"] for x in r], np.array([int(x["mc"]) for x in r]))
    Xp = lambda Fo, idx: mat(Fo, PRIMARY, idx)
    for tgt in a.targets.split(","):
        y = L[tgt]; T = dict(abs_target=ABS_TARGET.get(tgt)); res["targets"][tgt] = T
        print(f"\n== target {tgt}: range {y.min():.3f}..{y.max():.3f}", flush=True)
        T["constant"] = dict(val=rmse(y[va], y[tr].mean()), test=rmse(y[te], y[tr].mean()))
        rf = {}
        for name, cols in (("primary", PRIMARY), ("second", SECOND), ("truth", [c for c in TRUTH if c in F]), ("no_oop", NO_OOP)):
            if not cols:
                continue
            m = RandomForestRegressor(500, random_state=0, n_jobs=a.rf_jobs).fit(mat(F, cols, tr), y[tr])
            rf[name] = m
            T[f"rf_{name}"] = dict(val=rmse(y[va], m.predict(mat(F, cols, va))), test=rmse(y[te], m.predict(mat(F, cols, te))),
                                   importance=dict(zip(cols, np.round(m.feature_importances_, 3).tolist())))
        iso = {}
        for c in PRIMARY:
            m = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(F[c][tr], y[tr])
            iso[c] = m; T.setdefault("one_ratio", {})[c] = dict(val=rmse(y[va], m.predict(F[c][va])), test=rmse(y[te], m.predict(F[c][te])))
        best1 = min(T["one_ratio"], key=lambda c: T["one_ratio"][c]["val"]); T["one_ratio_best"] = best1
        print(f"  constant {T['constant']['val']:.4f} | RF primary {T['rf_primary']['val']:.4f}, total-normalised {T['rf_second']['val']:.4f}"
              + (f", noise-free window ratios {T['rf_truth']['val']:.4f}" if "rf_truth" in T else "")
              + f", without 12.0/12.7/13.5 {T['rf_no_oop']['val']:.4f}"
              + f" | best one-ratio (isotonic) {best1} {T['one_ratio'][best1]['val']:.4f}   [validation RMSE]", flush=True)
        print("  RF importances:", T["rf_primary"]["importance"], flush=True)
        chosen = {}
        if a.pysr:
            runs, T["pysr_log"] = pysr_runs(mat(F, PRIMARY, tr), y[tr], PRIMARY, a.out, tgt, a.niter, a.procs)
            T["deterministic_reproduces"] = ([r["equation"] for r in runs["det"]] == [r["equation"] for r in runs["det_repeat"]]
                                             if "det" in runs and "det_repeat" in runs else None)
            cand = []
            for name, fr in runs.items():
                for r in fr:
                    r["val"] = rmse(y[va], evalf(r["f"], Xp(F, va))) if np.all(np.isfinite(evalf(r["f"], Xp(F, va)))) else float("inf")
                    r["run"] = name
                    if name not in ("shuffled", "det_repeat"):
                        cand.append(r)
            T["shuffled_best_val"] = min(r["val"] for r in runs["shuffled"]) if "shuffled" in runs else None
            T["fronts"] = {name: [dict(complexity=r["complexity"], val=r["val"], equation=r["equation"]) for r in fr] for name, fr in runs.items()}
            for label, cmax in (("compact", 7), ("full", 12)):
                c = min((r for r in cand if r["complexity"] <= cmax), key=lambda r: r["val"])
                chosen[label] = c
                T[f"formula_{label}"] = dict(equation=c["equation"], complexity=c["complexity"], run=c["run"], val=c["val"],
                                             variables=used_vars(c["equation"], PRIMARY))
            print(f"  deterministic run reproduces: {T['deterministic_reproduces']}; shuffled-label best val {T['shuffled_best_val']} "
                  f"(constant {T['constant']['val']:.4f}); crashes per run: { {k: v['crashes'] for k, v in T['pysr_log'].items()} }", flush=True)
        # ---- final pass on the test set (after every selection)
        for label, c in chosen.items():
            d = T[f"formula_{label}"]; d["test"] = rmse(y[te], evalf(c["f"], Xp(F, te)))
            v = d["variables"]
            if len(v) == 1:                                                           # monotone in its single ratio?
                j = PRIMARY.index(v[0]); g = np.tile(np.median(Xp(F, tr), 0), (200, 1))
                g[:, j] = np.linspace(*np.percentile(F[v[0]][tr], [1, 99]), 200); dy = np.diff(evalf(c["f"], g))
                d["monotone"] = bool(np.all(dy >= 0) or np.all(dy <= 0))
            d["vs_rf"] = d["test"] / T["rf_primary"]["test"]; d["vs_one_ratio"] = d["test"] / T["one_ratio"][best1]["test"]
            print(f"  formula {label}: complexity {d['complexity']}, val {d['val']:.4f}, test {d['test']:.4f} "
                  f"({d['vs_rf']:.2f}x RF, {d['vs_one_ratio']:.2f}x one-ratio)   {d['equation']}", flush=True)
        with open(os.path.join(a.out, f"pred_{tgt}.csv"), "w", newline="") as fo:   # held-out rows, for the figures
            w = csv.writer(fo); w.writerow(["pop", "split", "y", "rf", f"one_ratio_{best1}"] + list(chosen))
            for sp, ix in (("val", va), ("test", te)):
                cols = [y[ix], rf["primary"].predict(Xp(F, ix)), iso[best1].predict(F[best1][ix])] + [evalf(c["f"], Xp(F, ix)) for c in chosen.values()]
                w.writerows([[int(i), sp] + [f"{c[k]:.5g}" for c in cols] for k, i in enumerate(ix)])
        print(f"  test RMSE: constant {T['constant']['test']:.4f}, RF {T['rf_primary']['test']:.4f}, one-ratio {best1} "
              f"{T['one_ratio'][best1]['test']:.4f}; target {T['abs_target']}", flush=True)
        # ---- controls on other feature tables: same populations, test rows that are ok in both
        for name, (Fo, pops) in other.items():
            if name == "clean":
                continue
            oko = np.zeros(n, bool); oko[pops] = Fo["ok"] & np.all(np.isfinite(Xp(Fo, slice(None))), 1)
            pos = {p: k for k, p in enumerate(pops)}
            ev = np.array([i for i in np.concatenate([va, te]) if oko[i]]); ko = np.array([pos[i] for i in ev])
            d = dict(n=len(ev), rf_main=rmse(y[ev], rf["primary"].predict(Xp(F, ev))), rf_other=rmse(y[ev], rf["primary"].predict(Xp(Fo, ko))),
                     rf_bias=float(np.mean(rf["primary"].predict(Xp(Fo, ko)) - y[ev])))
            for label, c in chosen.items():
                d[f"{label}_main"] = rmse(y[ev], evalf(c["f"], Xp(F, ev))); po = evalf(c["f"], Xp(Fo, ko))
                d[f"{label}_other"] = rmse(y[ev], po); d[f"{label}_bias"] = float(np.mean(po - y[ev]))
            T[f"control_{name}"] = d
            print(f"  control {name} (n={len(ev)} held-out rows): RF {d['rf_main']:.4f} -> {d['rf_other']:.4f} (bias {d['rf_bias']:+.4f})"
                  + "".join(f"; {l} {d[f'{l}_main']:.4f} -> {d[f'{l}_other']:.4f} (bias {d[f'{l}_bias']:+.4f})" for l in chosen), flush=True)
        # ---- Orion templates: application, not a gate
        if orion:
            Fo, tn, mc = orion; Xo = Xp(Fo, slice(None))
            mu, sd = Xp(F, tr).mean(0), Xp(F, tr).std(0); Z = (Xp(F, tr) - mu) / sd
            nn = NearestNeighbors(n_neighbors=6).fit(Z); d5 = nn.kneighbors(Z)[0][:, 5]; thr = float(np.percentile(d5, 99))
            do = nn.kneighbors((Xo - mu) / sd, n_neighbors=5)[0][:, 4]
            O = {}
            for t in dict.fromkeys(tn):
                m0 = np.array([x == t and k == 0 for x, k in zip(tn, mc)]); mm = np.array([x == t and k > 0 for x, k in zip(tn, mc)])
                e = dict(fit_ok=bool(Fo["ok"][m0][0]), knn_dist=float(do[m0][0]), knn_threshold=thr, extrapolation=bool(do[m0][0] > thr),
                         features={c: float(Fo[c][m0][0]) for c in PRIMARY},
                         features_train_1_99={c: np.percentile(F[c][tr], [1, 99]).round(4).tolist() for c in PRIMARY},
                         rf=float(rf["primary"].predict(Xo[m0])[0]), rf_mc_sd=float(np.std(rf["primary"].predict(Xo[mm]))) if mm.any() else None)
                for label, c in chosen.items():
                    p = evalf(c["f"], Xo); e[label] = float(p[m0][0]); e[f"{label}_mc_sd"] = float(np.nanstd(p[mm])) if mm.any() else None
                    e[f"{label}_in_label_range"] = bool(y.min() <= p[m0][0] <= y.max())
                O[t] = e
            T["orion"] = O
            print("  Orion templates (kNN distance / threshold; prediction):")
            for t, e in O.items():
                print(f"    {t:5s} dist {e['knn_dist']:.2f}/{thr:.2f} {'EXTRAPOLATION' if e['extrapolation'] else 'in domain   '} RF {e['rf']:.3f}"
                      + "".join(f"  {l} {e[l]:.3f} +- {e[f'{l}_mc_sd']:.3f}" for l in chosen), flush=True)
    for T in res["targets"].values():
        T.pop("fronts_f", None)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1, default=float)
    print(f"\nwrote {a.out}/results.json, features.csv" + (", front_*.csv" if a.pysr else ""))


if __name__ == "__main__":
    main()
