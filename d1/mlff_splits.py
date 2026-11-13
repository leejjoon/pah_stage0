# Train / validation / final-test splits for the charged-state MLFF seed set (committee review M3;
# docs/plan/charged_mlff_plan.md), plus a 61-100 C size-transfer test set.
#   python mlff_splits.py <seed dir> <pahdb.xml> [--frac 0.7,0.15,0.15] [--anchors 330,265,273,334,18,4,37]
#        [--transfer 2] [--seed 0]
# Reads <seed dir>/species.csv and xyz/ (from mlff_species.py), writes <seed dir>/species_v2.csv (the v1 file is kept
# as species_v1.csv) and the transfer pairs' geometries into xyz/.
# - Pairs are grouped by carbon skeleton (bond-graph hash of the carbon atoms only, from the neutral geometry), so a
#   skeleton and its hydrogenated or dehydrogenated variants never sit in different splits.
# - The lab-anchor benchmark species (--anchors, neutral PAHdb uids) go to final_test: the models never see them.
# - Every other skeleton group is assigned, within each 10-carbon size bin, to train / validation / final_test in
#   the --frac proportions (counted in pairs, groups shuffled with --seed).
# - transfer_test: --transfer pairs per 10-C bin from 61 to 100 C (farthest-point sampling in edge-structure fractions),
#   test only; sampled with fewer frames (see the plan) because a 100 C cation costs ~20 core-h per 6-31G* point.
import argparse, csv, hashlib, os, shutil, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlff_species import graph_hash, read_pahdb, SYM  # noqa: E402

FIELDS = ["pair", "formula", "nC", "split", "skeleton", "uid_q0", "uid_q1", "uid_qm1", "uid_q2", "mult_q2",
          "basis_q0", "basis_q1", "n_solo", "n_duo", "n_trio", "n_quartet"]


def read_xyz(p):
    L = open(p).read().split("\n"); n = int(L[0])
    rows = [l.split() for l in L[2:2 + n]]
    return [{"H": 1, "C": 6}[r[0]] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def skeleton(Z, P):
    c = [k for k, z in enumerate(Z) if z == 6]
    return graph_hash([6] * len(c), P[c])[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_dir"); ap.add_argument("xml")
    ap.add_argument("--frac", default="0.7,0.15,0.15"); ap.add_argument("--anchors", default="330,265,273,334,18,4,37")
    ap.add_argument("--transfer", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    D = a.seed_dir; fr = [float(x) for x in a.frac.split(",")]; anchors = set(a.anchors.split(","))
    src = f"{D}/species_v1.csv" if os.path.exists(f"{D}/species_v1.csv") else f"{D}/species.csv"
    if not os.path.exists(f"{D}/species_v1.csv"):
        shutil.copy(f"{D}/species.csv", f"{D}/species_v1.csv")
    rows = list(csv.DictReader(open(src)))
    for r in rows:
        r["skeleton"] = skeleton(*read_xyz(f"{D}/xyz/{r['pair']}_q0.xyz"))
    groups = defaultdict(list)
    for r in rows:
        groups[r["skeleton"]].append(r)
    rng = np.random.default_rng(a.seed)
    for g in groups.values():                                  # a group with an anchor goes to final_test whole
        if any(r["uid_q0"] in anchors for r in g):
            for r in g:
                r["split"] = "final_test"
    for lo in range(10, 70, 10):
        gs = [g for g in groups.values() if lo <= int(g[0]["nC"]) < lo + 10 and g[0].get("split") != "final_test"
              and not any(r["uid_q0"] in anchors for r in g)]
        gs = [gs[i] for i in rng.permutation(len(gs))]
        n_tot = sum(len(g) for g in gs) + sum(1 for r in rows if lo <= int(r["nC"]) < lo + 10 and r.get("split") == "final_test")
        target = {"validation": fr[1] * n_tot, "final_test": fr[2] * n_tot}
        have = {"validation": 0, "final_test": sum(1 for r in rows if lo <= int(r["nC"]) < lo + 10 and r.get("split") == "final_test")}
        for g in gs:
            s = next((k for k in ("final_test", "validation") if have[k] + len(g) <= target[k] + 0.5), "train")
            for r in g:
                r["split"] = s
            if s != "train":
                have[s] += len(g)
    # size-transfer pairs, 61-100 C
    S = read_pahdb(a.xml); by = defaultdict(dict)
    for s in S:
        s["hash"] = graph_hash(s["Z"], s["P"]); by[s["hash"]].setdefault(s["charge"], s)
    used = {r["skeleton"] for r in rows}
    pool = []
    for h, d in by.items():
        n, c = d.get(0), d.get(1)
        if n and c and n["method"] == "RB3LYP" and c["method"] == "UB3LYP" and 60 < n["nC"] <= 100:
            sk = skeleton(n["Z"], n["P"])
            if sk not in used:
                pool.append(dict(n=n, c=c, a=d.get(-1), d=d.get(2), skeleton=sk))
    feat = lambda p: np.array([p["n"]["edges"][k] for k in ("solo", "duo", "trio", "quartet")], float) / max(1, sum(p["n"]["edges"].values()))
    added = []
    for lo in range(61, 101, 10):
        cand = [p for p in pool if lo <= p["n"]["nC"] < lo + 10]
        sel = []
        while cand and len(sel) < a.transfer:
            if not sel:
                k = int(rng.integers(len(cand)))
            else:
                F = np.array([feat(p) for p in cand]); G = np.array([feat(p) for p in sel])
                k = int(np.argmax(np.linalg.norm(F[:, None] - G[None], axis=-1).min(1)))
            sel.append(cand.pop(k))
        added += sel
    for p in added:
        name = f"{p['n']['formula']}_{p['n']['uid']}"
        for q, r in ((0, p["n"]), (1, p["c"]), (-1, p["a"]), (2, p["d"])):
            if r is None:
                continue
            with open(f"{D}/xyz/{name}_q{q}.xyz", "w") as g:
                g.write(f"{len(r['Z'])}\nsource=PAHdb_v4.00 uid={r['uid']} charge={q} method={r['method']}/{r['basis']}\n")
                g.writelines(f"{SYM[z]} {x:.6f} {y:.6f} {zz:.6f}\n" for z, (x, y, zz) in zip(r["Z"], r["P"]))
        e = p["n"]["edges"]
        rows.append(dict(pair=name, formula=p["n"]["formula"], nC=p["n"]["nC"], split="transfer_test", skeleton=p["skeleton"],
                         uid_q0=p["n"]["uid"], uid_q1=p["c"]["uid"], uid_qm1=p["a"]["uid"] if p["a"] else "",
                         uid_q2=p["d"]["uid"] if p["d"] else "",
                         mult_q2=(p["d"]["state"].split("-")[0] if p["d"]["state"][:1].isdigit() else "?") if p["d"] else "",
                         basis_q0=p["n"]["basis"], basis_q1=p["c"]["basis"], n_solo=e["solo"], n_duo=e["duo"],
                         n_trio=e["trio"], n_quartet=e["quartet"]))
    with open(f"{D}/species_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} pairs -> {D}/species_v2.csv")
    for lo in list(range(10, 70, 10)) + [61, 71, 81, 91]:
        hi = lo + 10 if lo < 61 else lo + 9
        c = defaultdict(int)
        for r in rows:
            if lo <= int(r["nC"]) < hi + (1 if lo >= 61 else 0) and not (lo == 60 and False):
                c[r["split"]] += 1
        if c:
            print(f"  {lo:3d}-{hi - (0 if lo >= 61 else 1):3d} C: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    tot = defaultdict(int)
    for r in rows:
        tot[r["split"]] += 1
    print("  total: " + ", ".join(f"{k} {v}" for k, v in sorted(tot.items())),
          f"; skeleton groups with more than one pair: {sum(1 for g in groups.values() if len(g) > 1)}")


if __name__ == "__main__":
    main()
