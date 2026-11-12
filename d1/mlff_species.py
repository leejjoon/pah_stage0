# Seed species for charged-state MLFF training data (docs/plan/charged_mlff_plan.md): neutral/cation pairs of the
# same structure from PAHdb v4.00, C/H only, closed-shell neutral (RB3LYP) and doublet cation (UB3LYP), up to --cmax
# carbons. Pairs are matched by a bond-graph hash of the PAHdb geometries (each charge state has its own optimised
# geometry). The anion and dication of the same structure are recorded for later phases.
# Selection: per size bin (10-19, 20-29, ... C), farthest-point sampling in edge-structure fractions (solo, duo,
# trio, quartet), after the --must uids (neutral uids); then a random --holdout fraction of the rest is held out.
#   python mlff_species.py <pahdb.xml> <outdir> [--n 120] [--cmax 60] [--holdout 0.2] [--seed 0] [--must 330,265,...]
# Writes <outdir>/species.csv and <outdir>/xyz/<pair>_q<charge>.xyz (PAHdb geometry, uid and level in the comment).
import argparse, csv, hashlib, os, re
from collections import defaultdict
from xml.etree.ElementTree import iterparse
import numpy as np

RCOV = {1: 0.31, 6: 0.76}
SYM = {1: "H", 6: "C"}
L = lambda e: e.tag.split("}")[-1]


def kid(e, t):
    for k in e:
        if L(k) == t:
            return k


def graph_hash(Z, P, rounds=5):
    """Weisfeiler-Lehman hash of the element-labelled bond graph (distance < 1.2 x sum of covalent radii)."""
    r = np.array([RCOV[z] for z in Z]); d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    adj = (d < 1.2 * (r[:, None] + r[None])) & ~np.eye(len(Z), dtype=bool)
    lab = [str(z) for z in Z]
    for _ in range(rounds):
        lab = [hashlib.md5((lab[i] + "|" + ",".join(sorted(lab[j] for j in np.nonzero(adj[i])[0]))).encode()).hexdigest()[:12]
               for i in range(len(Z))]
    return hashlib.md5(",".join(sorted(lab)).encode()).hexdigest()


def read_pahdb(path):
    out = []
    for _, el in iterparse(open(path, "rb"), events=("end",)):
        if L(el) != "specie":
            continue
        atoms = [(int(kid(a, "type").text), [float(kid(a, c).text) for c in "xyz"]) for a in kid(el, "geometry")]
        Z = [z for z, _ in atoms]
        if set(Z) == {1, 6}:                               # C and H, both present (no bare carbon clusters)
            com = " ".join((x.text or "") for x in kid(el, "comments")).lower()
            m = re.search(r"b3lyp/([\w\-\+\*]+)", com)
            out.append(dict(uid=int(el.get("uid")), formula=kid(el, "formula").text, charge=int(kid(el, "charge").text),
                            method=kid(el, "method").text, state=(kid(el, "symmetry").text or "").strip(), basis=m.group(1) if m else "?", Z=Z,
                            P=np.array([p for _, p in atoms]), nC=Z.count(6),
                            edges={k: int(kid(el, f"n_{k}").text) for k in ("solo", "duo", "trio", "quartet")}))
        el.clear()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--cmax", type=int, default=60)
    ap.add_argument("--holdout", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--must", default="330,265,273,334,18,4,37", help="neutral uids always in the training split")
    a = ap.parse_args()
    S = read_pahdb(a.xml)
    by = defaultdict(dict)                                   # graph hash -> charge -> record (first one wins)
    for s in S:
        s["hash"] = graph_hash(s["Z"], s["P"])
        by[s["hash"]].setdefault(s["charge"], s)
    pairs = []
    for h, d in by.items():
        n, c = d.get(0), d.get(1)
        if n and c and n["method"] == "RB3LYP" and c["method"] == "UB3LYP" and n["nC"] <= a.cmax:
            pairs.append(dict(hash=h, n=n, c=c, a=d.get(-1), d=d.get(2)))
    print(f"{len(S)} C/H species; {len(pairs)} closed-shell neutral / doublet cation pairs with <= {a.cmax} C")
    must = {int(u) for u in a.must.split(",") if u}
    chosen = [p for p in pairs if p["n"]["uid"] in must]
    rest = [p for p in pairs if p["n"]["uid"] not in must]
    feat = lambda p: np.array([p["n"]["edges"][k] for k in ("solo", "duo", "trio", "quartet")], float) / max(1, sum(p["n"]["edges"].values()))
    bins = list(range(10, a.cmax, 10))                       # last bin includes cmax
    inb = lambda p, lo: lo <= p["n"]["nC"] < lo + 10 or (lo == bins[-1] and lo <= p["n"]["nC"] <= a.cmax)
    quota = max(1, (a.n - len(chosen)) // len(bins))
    for lo in bins:
        pool = [p for p in rest if inb(p, lo)]
        sel = [p for p in chosen if inb(p, lo)]
        rng = np.random.default_rng(a.seed + lo)
        while pool and len(sel) < quota + sum(inb(p, lo) for p in chosen):
            if not sel:
                k = int(rng.integers(len(pool)))
            else:
                F = np.array([feat(p) for p in pool]); G = np.array([feat(p) for p in sel])
                dist = np.linalg.norm(F[:, None] - G[None], axis=-1).min(1) + 0.01 * np.abs(
                    np.array([p["n"]["nC"] for p in pool]) - np.mean([p["n"]["nC"] for p in sel])) / 10
                k = int(np.argmax(dist))
            sel.append(pool.pop(k))
        chosen += [p for p in sel if p not in chosen]
        print(f"  {lo:3d}-{lo + 9 + (lo == bins[-1]):3d} C: {sum(inb(p, lo) for p in pairs)} pairs available, "
              f"{sum(inb(p, lo) for p in chosen)} chosen")
    rng = np.random.default_rng(a.seed)
    free = [i for i, p in enumerate(chosen) if p["n"]["uid"] not in must]
    held = set(rng.choice(free, size=int(round(a.holdout * len(chosen))), replace=False).tolist()) if free else set()
    os.makedirs(f"{a.out}/xyz", exist_ok=True)
    with open(f"{a.out}/species.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "formula", "nC", "split", "uid_q0", "uid_q1", "uid_qm1", "uid_q2", "mult_q2",
                    "basis_q0", "basis_q1", "n_solo", "n_duo", "n_trio", "n_quartet"])
        for i, p in enumerate(sorted(chosen, key=lambda p: (p["n"]["nC"], p["n"]["uid"]))):
            name = f"{p['n']['formula']}_{p['n']['uid']}"
            for q, r in ((0, p["n"]), (1, p["c"]), (-1, p["a"]), (2, p["d"])):
                if r is None:
                    continue
                with open(f"{a.out}/xyz/{name}_q{q}.xyz", "w") as g:
                    g.write(f"{len(r['Z'])}\nPAHdb v4.00 uid {r['uid']} charge {q} {r['method']}/{r['basis']}\n")
                    g.writelines(f"{SYM[z]} {x:.6f} {y:.6f} {zz:.6f}\n" for z, (x, y, zz) in zip(r["Z"], r["P"]))
            split = "holdout" if chosen.index(p) in held else "train"
            e = p["n"]["edges"]
            w.writerow([name, p["n"]["formula"], p["n"]["nC"], split, p["n"]["uid"], p["c"]["uid"],
                        p["a"]["uid"] if p["a"] else "", p["d"]["uid"] if p["d"] else "",
                        (p["d"]["state"].split("-")[0] if p["d"]["state"][:1].isdigit() else "?") if p["d"] else "",
                        p["n"]["basis"], p["c"]["basis"], e["solo"], e["duo"], e["trio"], e["quartet"]])
    print(f"{len(chosen)} pairs ({len(held)} held out) -> {a.out}/species.csv")


if __name__ == "__main__":
    main()
