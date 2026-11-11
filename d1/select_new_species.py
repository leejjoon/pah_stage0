"""Choose new D1 MLMD species from PAHdb v4.00 that fill gaps left by Mai et al.'s 1,704
(plan: docs/plan/d1_hangang_new_species.md, step 1). Standard library only. Deterministic.

  python3 d1/select_new_species.py PAHDB.xml.gz MAI_XYZ_DIR OUTDIR [--n 1000 --npah 100] [--write-xyz XYZ_OUT]

Pool: neutral species with only H, C and N atoms, one connected molecule, Nc >= 10, at most 300 atoms, inside the
structure range of Mai's set (in_domain: no pure-carbon clusters), not in Mai's set (matched by structure: formula plus a Weisfeiler-Lehman hash of the bond graph, so UIDs are not
trusted). Duplicates of one structure in v4 (other electronic states) keep the lowest total energy.
Cells: Nc bin x dominant H adjacency x class. Greedy fill: repeatedly add a species to the cell with the
lowest (Mai + chosen) / weight, among cells with candidates left; inside a cell, the least-used formula first
(at most --max-per-formula new species per formula), then Nc spread. N-PAHs are filled separately (--npah).
Writes OUTDIR/new_species_v1.csv, OUTDIR/cells_v1.csv and, with --write-xyz, one xyz per species in Mai's
format, named <formula>_v4u<uid>.xyz.
"""
import sys, os, gzip, csv, argparse, collections, hashlib
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_xyz import profile_atoms, bonds, read  # noqa: E402

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
BINS = [(10, 20), (20, 30), (30, 40), (40, 60), (60, 80), (80, 100), (100, 130), (130, 400)]
W_BIN = [0.5, 0.75, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5]          # weight new species to Nc 40-200
ADJ = ("solo", "duo", "trio", "quartet")
W_ADJ = {"solo": 1.5, "duo": 1.5, "trio": 1.0, "quartet": 1.0}  # and to compact (solo/duo-rich) edges
W_CLS = {"benzenoid": 1.0, "superH": 0.5, "dehydro": 0.5, "pentagon": 0.3, "npah": 1.0}
MAX_PER_FORMULA = 40               # PAHdb v4 additions are isomer sets; 40 reaches 1,000 (20 gives 828)
MAX_ATOMS = 300


def in_domain(p):
    """Inside the structure range of Mai's 1,704 (a proxy for the model's training domain): has H (pure-carbon
    clusters and cages are out), at most 2 bare edge C, 2 pentagons, 12 sp3 C."""
    return p["nH"] > 0 and p["bareC"] <= 2 and p["pent"] <= 2 and p["sp3C"] <= 12


def binof(nc):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= nc < hi:
            return i
    return None


def classify(p):
    return ("npah" if p["nN"] else "pentagon" if p["pent"] else "superH" if p["sp3C"] or p["CH2"] else
            "dehydro" if p["bareC"] else "benzenoid")


def dominant(p):
    return max(ADJ, key=lambda k: (p[k], -ADJ.index(k)))


def wl_hash(at, nb, rounds=4):
    """Weisfeiler-Lehman hash of the bond graph; same for any atom order. Plain 1-WL with element labels cannot
    tell many benzenoid isomers apart (59 false collisions among Mai's 1,704), so each atom's starting label also
    holds its graph-distance profile (how many atoms of each element at each bond distance)."""
    lab = []
    for i in range(len(at)):
        dist, front = {i: 0}, [i]
        while front:
            nxt = []
            for v in front:
                for w in nb[v]:
                    if w not in dist:
                        dist[w] = dist[v] + 1; nxt.append(w)
            front = nxt
        prof = collections.Counter((at[j][0], d) for j, d in dist.items())
        lab.append(at[i][0] + ":" + ",".join("%s%d*%d" % (e, d, c) for (e, d), c in sorted(prof.items())))
    for _ in range(rounds):
        lab = [hashlib.sha1((lab[i] + "|" + ",".join(sorted(lab[j] for j in nb[i]))).encode()).hexdigest()[:16]
               for i in range(len(at))]
    return hashlib.sha1(",".join(sorted(lab)).encode()).hexdigest()[:20]


def formula(at):
    c = collections.Counter(a[0] for a in at)
    return "".join(e + (str(c[e]) if c[e] > 1 else "") for e in ("C", "H", "N", "O") if c[e])


def parse(xml_gz):
    """Yield (uid, formula, charge, symmetry, total_e, atoms) for every species."""
    with gzip.open(xml_gz) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag != NS + "specie":
                continue
            g = el.find(NS + "geometry")
            atoms = []
            for a in g.findall(NS + "atom"):
                z = int(a.find(NS + "type").text)
                atoms.append((SYM.get(z, "X%d" % z), float(a.find(NS + "x").text), float(a.find(NS + "y").text),
                              float(a.find(NS + "z").text)))
            te = el.find(NS + "total_e")
            yield (int(el.get("uid")), el.find(NS + "formula").text, int(el.find(NS + "charge").text),
                   el.find(NS + "symmetry").text, float(te.text) if te is not None else 0.0, atoms)
            el.clear()


def spread_pick(cands, used_formula, chosen_nc):
    """Least-used formula first, then the Nc farthest from species already chosen in this cell, then UID."""
    def key(c):
        d = min([abs(c["nC"] - n) for n in chosen_nc] or [999])
        return (used_formula[c["formula"]], -d, c["uid"])
    return min(cands, key=key)


def fill(pool, base_count, n, cells_of, cap):
    """Greedy fill of n species from pool (dicts with 'cell'); base_count: Mai species per cell."""
    left = collections.defaultdict(list)
    for c in pool:
        left[c["cell"]].append(c)
    chosen, used = [], collections.Counter()
    new_count = collections.Counter()
    nc_in = collections.defaultdict(list)
    while len(chosen) < n:
        open_cells = [k for k, v in left.items() if any(used[c["formula"]] < cap for c in v)]
        if not open_cells:
            break
        k = min(open_cells, key=lambda k: ((base_count[k] + new_count[k]) / cells_of(k), k))
        cands = [c for c in left[k] if used[c["formula"]] < cap]
        c = spread_pick(cands, used, nc_in[k])
        left[k].remove(c); chosen.append(c); used[c["formula"]] += 1; new_count[k] += 1; nc_in[k].append(c["nC"])
    return chosen, new_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml"); ap.add_argument("mai_xyz"); ap.add_argument("outdir")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--npah", type=int, default=100)
    ap.add_argument("--write-xyz", default=None)
    ap.add_argument("--max-per-formula", type=int, default=MAX_PER_FORMULA)
    a = ap.parse_args()

    # Mai's set: profiles and structure keys
    mai = {}
    for fn in sorted(os.listdir(a.mai_xyz)):
        if fn.endswith(".xyz"):
            at = read(os.path.join(a.mai_xyz, fn))
            p = profile_atoms(at, fn)
            p["key"] = formula(at) + ":" + wl_hash(at, bonds(at))
            mai[fn[:-4]] = p
    mai_keys = {p["key"]: s for s, p in mai.items()}
    print("Mai: %d species, %d distinct structure keys" % (len(mai), len(mai_keys)))

    # PAHdb v4.00 neutrals
    stats = collections.Counter()
    best = {}
    matched_mai = {}
    for uid, form, charge, sym, te, at in parse(a.xml):
        stats["species"] += 1
        if charge != 0:
            continue
        stats["neutral"] += 1
        if any(s not in ("H", "C", "N") for s, *_ in at):
            stats["neutral, other elements"] += 1
            continue
        p = profile_atoms(at, "%s_v4u%d.xyz" % (form, uid))
        if p["frags"] != 1:
            stats["neutral CHN, >1 fragment (cluster)"] += 1
            continue
        key = formula(at) + ":" + wl_hash(at, bonds(at))
        if key in mai_keys:
            stats["neutral CHN, in Mai's set"] += 1
            matched_mai[mai_keys[key]] = uid
            continue
        if not in_domain(p):
            stats["neutral CHN, outside Mai's structure range"] += 1
            continue
        if p["nC"] < 10 or len(at) > MAX_ATOMS:
            stats["neutral CHN, Nc < 10 or > %d atoms" % MAX_ATOMS] += 1
            continue
        p.update(uid=uid, formula=form, symmetry=sym, total_e=te, natoms=len(at), key=key, atoms=at)
        if key in best:
            stats["neutral CHN, duplicate structure"] += 1
            if te >= best[key]["total_e"]:
                continue
        best[key] = p
    for k, v in stats.items():
        print("  %-40s %d" % (k, v))
    print("  Mai species found in v4 by structure: %d of %d (same UID: %d)" % (
        len(matched_mai), len(mai), sum(1 for s, u in matched_mai.items() if s.split("_")[-1] == str(u))))

    pool = list(best.values())
    for p in list(mai.values()) + pool:
        p["cls"], p["dom"], p["bin"] = classify(p), dominant(p), binof(p["nC"])
        p["cell"] = (p["bin"], p["dom"], p["cls"])
    pool = [p for p in pool if p["bin"] is not None]
    print("pool: %d candidates (%d N-PAH)" % (len(pool), sum(p["cls"] == "npah" for p in pool)))

    base = collections.Counter(p["cell"] for p in mai.values() if p["bin"] is not None)
    weight = lambda k: W_BIN[k[0]] * W_ADJ[k[1]] * W_CLS[k[2]]
    ch_n, cnt_n = fill([p for p in pool if p["cls"] == "npah"], base, a.npah, weight, a.max_per_formula)
    ch_c, cnt_c = fill([p for p in pool if p["cls"] != "npah"], base, a.n - len(ch_n), weight, a.max_per_formula)
    chosen = sorted(ch_c + ch_n, key=lambda p: (p["nC"], p["uid"]))
    new = cnt_n + cnt_c
    poolc = collections.Counter(p["cell"] for p in pool)
    print("chosen: %d (%d N-PAH), %d atoms in all" % (len(chosen), len(ch_n), sum(p["natoms"] for p in chosen)))

    os.makedirs(a.outdir, exist_ok=True)
    keys = ["species", "uid", "formula", "symmetry", "cls", "nC", "nH", "nN", "natoms", "dom",
            "solo", "duo", "trio", "quartet", "sp3C", "CH2", "bareC", "pent"]
    with open(os.path.join(a.outdir, "new_species_v1.csv"), "w") as f:
        w = csv.writer(f); w.writerow(keys)
        for p in chosen:
            p["species"] = "%s_v4u%d" % (p["formula"], p["uid"])
            w.writerow([p[k] for k in keys])
    cells = sorted(set(base) | set(poolc), key=lambda k: (k[2] != "benzenoid", k[2], k[0], ADJ.index(k[1])))
    with open(os.path.join(a.outdir, "cells_v1.csv"), "w") as f:
        w = csv.writer(f); w.writerow(["Nc_bin", "dominant", "class", "mai", "pool", "new", "weight"])
        for k in cells:
            lo, hi = BINS[k[0]]
            w.writerow(["%d-%d" % (lo, hi - 1), k[1], k[2], base[k], poolc[k], new[k], weight(k)])
    print("tables in", a.outdir)

    if a.write_xyz:
        os.makedirs(a.write_xyz, exist_ok=True)
        for p in chosen:
            with open(os.path.join(a.write_xyz, p["species"] + ".xyz"), "w") as f:
                f.write("%d\n%d    %s\n" % (p["natoms"], p["uid"], p["formula"]))
                for s, x, y, z in p["atoms"]:
                    f.write("%s %.6f %.6f %.6f\n" % (s, x, y, z))
        print("xyz files in", a.write_xyz)


if __name__ == "__main__":
    main()
