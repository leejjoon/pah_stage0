# Species for the D1 cation finite-T rules (d1/tdep_rules_cation_v1; coordinator 2026-11-26 08:58):
#   python cation_mlmd_select.py <species_nc_v1.csv> <pahdb_v4_exclude_nonkekule_v1.csv> <pahdb-v4.00.xml> <out.csv> <xyz dir> [--n 100]
# Candidates: slice/species_nc_v1.csv (gh200-containers) rows with charge 1, status "use" (the table's verdict column),
# nN = 0 and atoms C/H only, 26 <= Nc <= 120, doublet (a singlet cation has a radical parent), no sp3 carbon, and a
# Kekule carbon skeleton: the sp2 framework of the PAHdb cation geometry is bipartite with a perfect matching (the
# test of docs/reports/pahdb_v4_nonkekule/pahdb_nonkekule.py; a cation shares its parent neutral's skeleton, and every
# neutral on pahdb_v4_exclude_nonkekule_v1.csv failed that same test, so it subsumes the list; the list is also checked
# by formula as a cross-check). Stratified: 4 Nc bins (26-40, 41-60, 61-85, 86-120) x n/4 species; inside a bin the
# species are grouped by H-edge class (the class holding most H: solo, duo, trio, quartet; ties to the lower class),
# the bin's quota split as evenly as the classes allow (round robin), and inside a class the picks are spread evenly
# over Nc. Ovalene+ (C32H14, the smoke-gate species) is always included. Writes <out.csv> and <xyz dir>/<uid>_q1.xyz
# (the PAHdb B3LYP/6-31G* geometry, the MD start).
import argparse, csv, re, sys
import xml.etree.ElementTree as ET
import numpy as np

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
RCOV = {"H": 0.31, "C": 0.76}
BINS = [(26, 40), (41, 60), (61, 85), (86, 120)]
CLASSES = ["solo", "duo", "trio", "quartet"]
sys.setrecursionlimit(20000)


def neighbours(el, xyz):
    r = np.array([RCOV[e] for e in el]); d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    b = (d < 1.15 * (r[:, None] + r[None])) & ~np.eye(len(el), dtype=bool)
    return [list(np.nonzero(row)[0]) for row in b]


def kekule(el, nb):  # verbatim logic of pahdb_nonkekule.py: (has Kekule structure, imbalance) or (None, None)
    pi = [i for i, e in enumerate(el) if e == "C" and len(nb[i]) == 3]; s = set(pi)
    adj = {i: [j for j in nb[i] if j in s] for i in pi}; col = {}
    for st in pi:
        if st in col:
            continue
        col[st] = 0; stack = [st]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in col:
                    col[v] = 1 - col[u]; stack.append(v)
                elif col[v] == col[u]:
                    return None, None
    A = [i for i in pi if col[i] == 0]; B = [i for i in pi if col[i] == 1]; match = {}

    def augment(u, seen):
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                if v not in match or augment(match[v], seen):
                    match[v] = u; return True
        return False
    m = sum(augment(u, set()) for u in A)
    return len(A) == len(B) == m, len(A) - len(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species"); ap.add_argument("exclude"); ap.add_argument("xml"); ap.add_argument("out"); ap.add_argument("xyz_dir")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(a.species)) if r["charge"] == "1" and r["status"] == "use" and r["nN"] == "0"
            and 26 <= int(r["nC"]) <= 120]
    want = {r["uid"]: r for r in rows}
    ex_formula = {r["formula"] for r in csv.DictReader(open(a.exclude))}
    geo, why = {}, {}
    for _, sp in ET.iterparse(a.xml, events=("end",)):
        if sp.tag != NS + "specie":
            continue
        u = sp.get("uid")
        if u in want:
            atoms = sp.find(NS + "geometry").findall(NS + "atom")
            z = [int(x.find(NS + "type").text) for x in atoms]
            el = [{1: "H", 6: "C"}.get(k, "X") for k in z]
            xyz = np.array([[float(x.find(NS + t).text) for t in ("x", "y", "z")] for x in atoms])
            nb = neighbours(el, xyz) if "X" not in el else None
            k, imb = kekule(el, nb) if nb else (None, None)
            if "X" in el:
                why[u] = "atoms other than C/H"
            elif any(e == "C" and len(nb[i]) == 4 for i, e in enumerate(el)):
                why[u] = "sp3 carbon"
            elif int(want[u]["mult"]) != 2:
                why[u] = f"multiplicity {want[u]['mult']} (radical parent)"
            elif k is None:
                why[u] = "non-alternant skeleton"
            elif not k:
                why[u] = f"non-Kekule skeleton (imbalance {imb})"
            else:
                geo[u] = (el, xyz, imb)
        sp.clear()
    for u, w in sorted(why.items()):
        print(f"excluded {want[u]['species']}: {w}", file=sys.stderr)
    ok = [want[u] for u in geo]
    neutral_formula = lambda r: r["formula"].rstrip("+")
    print(f"{len(rows)} C/H use cations with 26 <= Nc <= 120; {len(ok)} Kekule doublets; "
          f"{sum(neutral_formula(r) in ex_formula for r in ok)} share a formula with an excluded neutral (isomers; their own skeleton is Kekule)")

    def hclass(r):
        h = [int(r[c]) for c in CLASSES]
        return CLASSES[int(np.argmax(h))]
    per = a.n // len(BINS); pick = []
    for lo, hi in BINS:
        b = [r for r in ok if lo <= int(r["nC"]) <= hi]
        by = {c: sorted([r for r in b if hclass(r) == c], key=lambda r: (int(r["nC"]), int(r["uid"]))) for c in CLASSES}
        quota = dict.fromkeys(CLASSES, 0); left = per
        while left:                                     # round robin over classes that still have species
            moved = False
            for c in CLASSES:
                if left and quota[c] < len(by[c]):
                    quota[c] += 1; left -= 1; moved = True
            if not moved:
                break
        for c in CLASSES:
            if quota[c]:
                idx = np.unique(np.round(np.linspace(0, len(by[c]) - 1, quota[c])).astype(int))
                pick += [dict(by[c][i], nc_bin=f"{lo}-{hi}", h_class=c) for i in idx]
        print(f"Nc {lo}-{hi}: {len(b)} candidates " + ", ".join(f"{c} {len(by[c])}->{quota[c]}" for c in CLASSES))
    ov = [r for r in ok if r["formula"] == "C32H14+"]
    if ov and not any(p["formula"] == "C32H14+" for p in pick):   # ovalene+: replace the nearest-Nc pick of its class
        o = dict(ov[0], nc_bin="26-40", h_class=hclass(ov[0]))
        same = [i for i, p in enumerate(pick) if p["nc_bin"] == "26-40" and p["h_class"] == o["h_class"]] or \
               [i for i, p in enumerate(pick) if p["nc_bin"] == "26-40"]
        pick[min(same, key=lambda i: abs(int(pick[i]["nC"]) - 32))] = o
    cols = ["species", "uid", "formula", "nC", "nH", "solo", "duo", "trio", "quartet", "nc_bin", "h_class", "imbalance", "xyz"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for p in sorted(pick, key=lambda r: (int(r["nC"]), int(r["uid"]))):
            el, xyz, imb = geo[p["uid"]]; name = f"{p['uid']}_q1"
            with open(f"{a.xyz_dir}/{name}.xyz", "w") as g:
                g.write(f"{len(el)}\n{p['species']} PAHdb v4.00 uid {p['uid']} charge 1 mult 2\n")
                g.write("".join(f"{e} {x:.8f} {y:.8f} {z:.8f}\n" for e, (x, y, z) in zip(el, xyz)))
            w.writerow(dict(p, imbalance=imb, xyz=f"{name}.xyz"))
    print(f"wrote {len(pick)} species to {a.out}")


if __name__ == "__main__":
    main()
