"""Choose the 48 overlap species from Mai et al.'s 1,704 (plan: docs/plan/d1_hangang_new_species.md, step 3).
Deterministic. 40 pure benzenoids (C/H only, no sp3 C, no bare edge C, no pentagon), 5 per Nc bin, spread over
dominant adjacency classes and Nc, distinct formulas; plus 8 special: 3 N-PAH, 2 pentagon, 2 superhydrogenated,
1 dehydrogenated. Naphthalene (C10H8_330, the step 2b species) is always included.
Usage (standard library only):
  python3 d1/profile_xyz.py /appsdata/$USER/d1/XYZ_pahdb_1704 mai1704_profile.csv
  python3 d1/select_overlap.py mai1704_profile.csv d1/overlap_species_v1.csv
"""
import sys, csv

BINS = [(10, 20), (20, 30), (30, 40), (40, 60), (60, 80), (80, 100), (100, 130), (130, 217)]
PER_BIN = 5
SPECIAL = [("npah", 3), ("pentagon", 2), ("superH", 2), ("dehydro", 1)]
ADJ = ("solo", "duo", "trio", "quartet")
ALWAYS = ["C10H8_330"]

rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    r["sp"] = r["file"][:-4]
    r["formula"] = r["sp"].rsplit("_", 1)[0]
    for k in ("nC", "nH", "nN", "nO", "other", "sp3C", "CH2", "bareC", "pent", "solo", "duo", "trio", "quartet"):
        r[k] = int(float(r[k]))
    r["dom"] = max(ADJ, key=lambda k: (r[k], -ADJ.index(k)))
    r["cls"] = ("npah" if r["nN"] else "pentagon" if r["pent"] else "superH" if r["sp3C"] else
                "dehydro" if r["bareC"] else "benzenoid")
rows.sort(key=lambda r: (r["nC"], r["nH"], int(r["uid"])))


def spread(cands, k, taken):
    """k candidates evenly spaced in the (sorted) list, distinct formulas, skipping those taken."""
    out, forms = [], {t["formula"] for t in taken}
    cands = [c for c in cands if c not in taken]
    if not cands:
        return out
    for i in range(k):
        j0 = round((i + 0.5) * len(cands) / k - 0.5)
        for j in sorted(range(len(cands)), key=lambda j: abs(j - j0)):
            c = cands[j]
            if c not in out and c["formula"] not in forms:
                out.append(c); forms.add(c["formula"]); break
    return out


chosen = [r for r in rows if r["sp"] in ALWAYS]
for lo, hi in BINS:
    pool = [r for r in rows if r["cls"] == "benzenoid" and lo <= r["nC"] < hi]
    picked = [r for r in chosen if r in pool]
    for a in ADJ:                                       # one per dominant adjacency class present
        if len(picked) >= PER_BIN or any(p["dom"] == a for p in picked):
            continue
        picked += spread([r for r in pool if r["dom"] == a], 1, chosen + picked)
    picked += spread(pool, PER_BIN - len(picked), chosen + picked)
    chosen += [p for p in picked if p not in chosen]
for cls, k in SPECIAL:
    chosen += spread([r for r in rows if r["cls"] == cls], k, chosen)

chosen.sort(key=lambda r: (r["nC"], r["sp"]))
keys = ["sp", "formula", "uid", "cls", "nC", "nH", "nN", "dom", "solo", "duo", "trio", "quartet", "sp3C", "bareC", "pent"]
with open(sys.argv[2], "w") as f:
    w = csv.writer(f); w.writerow(["species"] + keys[1:])
    for r in chosen:
        w.writerow([r[k] for k in keys])
natoms = sum(r["nC"] + r["nH"] + r["nN"] for r in chosen)
print(len(chosen), "species,", natoms, "atoms in all ->", sys.argv[2])
for r in chosen:
    print("  %-14s %-9s Nc %3d  %-8s solo/duo/trio/quartet %d/%d/%d/%d" % (r["sp"], r["cls"], r["nC"], r["dom"],
                                                                         r["solo"], r["duo"], r["trio"], r["quartet"]))
