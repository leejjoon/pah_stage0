# Kekule check of PAHdb v4.00 species and of D1 libraries (docs/plan/d1_hangang_new_species.md, "PAHdb intensity anomaly").
# A PAH whose sp2-carbon graph has no perfect matching (no Kekule structure) cannot be closed-shell; for an alternant
# (bipartite) graph the sublattice imbalance |nA - nB| = 2S gives the ground-state spin (Lieb's theorem: 2 -> triplet).
# PAHdb lists 239 such pure C/H neutrals as closed-shell singlets ("1-..."), and all have anomalous harmonic
# intensities (6x the per-carbon total of normal neutrals, CH-stretch share < 15 %).
#   python kekule_check.py pahdb <pahdb xml.gz> <out csv>      every species: uid, formula, charge, state, kekule, imbalance
#   python kekule_check.py xyz <xyz dir> <out csv>             every <species>.xyz in a directory
# kekule: 1 = has a Kekule structure, 0 = none, "" = non-alternant (odd rings; not tested). Standard library only.
import csv, gzip, os, sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_xyz import bonds, read  # noqa: E402

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 12: "Mg", 14: "Si", 26: "Fe"}
sys.setrecursionlimit(20000)


def kekule(at):
    """(has_kekule, imbalance) for the sp2 C/N framework (atoms with 3 neighbours), or (None, None) if not bipartite."""
    nb = bonds(at)
    pi = [i for i, a in enumerate(at) if a[0] in ("C", "N") and len(nb[i]) == 3]; s = set(pi)
    adj = {i: [j for j in nb[i] if j in s] for i in pi}
    col = {}
    for st in pi:
        if st in col:
            continue
        col[st] = 0; q = [st]
        while q:
            u = q.pop()
            for v in adj[u]:
                if v not in col:
                    col[v] = 1 - col[u]; q.append(v)
                elif col[v] == col[u]:
                    return None, None
    left = [i for i in pi if col[i] == 0]; right = [i for i in pi if col[i] == 1]; match = {}

    def aug(u, seen):
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                if v not in match or aug(match[v], seen):
                    match[v] = u; return True
        return False
    m = sum(aug(u, set()) for u in left)
    return int(len(left) == len(right) == m), len(left) - len(right)


def main(mode, src, out):
    rows = []
    if mode == "pahdb":
        with gzip.open(src) as f:
            for ev, el in ET.iterparse(f, events=("end",)):
                if el.tag != NS + "specie":
                    continue
                at = [(SYM.get(int(a.find(NS + "type").text), "X"), float(a.find(NS + "x").text), float(a.find(NS + "y").text),
                       float(a.find(NS + "z").text)) for a in el.find(NS + "geometry").findall(NS + "atom")]
                if set(a[0] for a in at) - {"C", "H", "N", "O"}:   # Mg/Si/Fe complexes: not handled
                    el.clear(); continue
                k, imb = kekule(at)
                rows.append(dict(uid=int(el.get("uid")), formula=el.find(NS + "formula").text, charge=int(el.find(NS + "charge").text),
                                 state=el.find(NS + "symmetry").text, kekule="" if k is None else k, imbalance="" if imb is None else imb))
                el.clear()
    else:
        for fn in sorted(os.listdir(src)):
            if fn.endswith(".xyz"):
                k, imb = kekule(read(os.path.join(src, fn)))
                rows.append(dict(species=fn[:-4], kekule="" if k is None else k, imbalance="" if imb is None else imb))
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} -> {out}; no Kekule structure: {sum(r['kekule'] == 0 for r in rows)}; non-alternant: {sum(r['kekule'] == '' for r in rows)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
