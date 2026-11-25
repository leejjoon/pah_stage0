# Species pool for the neutral + cation prototype, version 3 (coordinator request 2026-11-24).
#   python slice/nc_pool.py <pahdb xml or xml.gz> [out csv, default slice/species_nc_v1.csv]      (needs numpy)
# Rows:
#   neutrals  slice/neutral_species_v1.csv, unchanged (819 with status = use; neutral_pool.py)
#   cations   every PAHdb v4.00 species with charge +1 and 26-196 C. A cation is used when
#               - d1/screen_spin.py finds its listed multiplicity consistent with its bonding (verdict ok),
#               - its Gaussian route line names B3LYP with the 6-31G* basis (PAHdb's `method` gives R/UB3LYP only),
#               - it is a planar benzenoid or N-containing PAH by the same structure profile as the neutrals
#                 (d1/profile_xyz.py: one fragment, only H/C/N, no sp3 or CH2 carbon, no bare edge carbon, no
#                 pentagon; rms distance from the best plane < 0.05 A).
#             Everything else is kept with the reason in `status`.
# Cations take PAHdb's harmonic line lists. They are not paired with neutrals: structure_id is the cation's own
# uid, and populations.py --design v3 mixes a neutral and a cation sub-population. Columns as neutral_pool.py, plus
# `basis` (from the route line; empty for the neutral rows, which were not filtered on it).
import csv, gzip, os, re, sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); D1 = os.path.join(HERE, "..", "d1")
sys.path.insert(0, D1)
from profile_xyz import profile_atoms  # noqa: E402

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 12: "Mg", 14: "Si", 26: "Fe"}
COLS = ["species", "source", "uid", "structure_id", "formula", "charge", "mult", "nC", "nH", "nN", "cls", "solo", "duo", "trio",
        "quartet", "lib300", "lib600", "lib900", "status", "basis"]


def basis_of(route):
    r = route.lower().replace(" ", "")
    if "b3lyp" not in r:
        return "not B3LYP"
    m = re.search(r"b3lyp/([^\s]+)", route.lower())
    b = m.group(1) if m else "?"
    return "6-31G*" if b.startswith(("6-31g*", "6-31g(d)")) and not b.startswith(("6-31g**", "6-31g(d,p)")) else b


def main(xml, out):
    scr = {int(r["uid"]): r for r in csv.DictReader(open(os.path.join(D1, "pahdb_v4_screen_spin_v1.csv")))}
    rows = [dict(r, basis="") for r in csv.DictReader(open(os.path.join(HERE, "neutral_species_v1.csv")))]
    n0 = len(rows)
    with (gzip.open(xml) if xml.endswith(".gz") else open(xml, "rb")) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag != NS + "specie":
                continue
            uid = int(el.get("uid")); s = scr.get(uid)
            if int(el.find(NS + "charge").text) == 1 and s is not None and 26 <= int(s["nC"]) <= 196:
                at = [(SYM.get(int(a.find(NS + "type").text), "X"), float(a.find(NS + "x").text), float(a.find(NS + "y").text),
                       float(a.find(NS + "z").text)) for a in el.find(NS + "geometry").findall(NS + "atom")]
                route = " ".join(c.text or "" for c in el.iter(NS + "comment")); bas = basis_of(route)
                X = np.array([a[1:] for a in at]); X = X - X.mean(0); flat = float(np.linalg.svd(X, compute_uv=False)[-1] / np.sqrt(len(X)))
                if any(a[0] not in "HCN" for a in at):
                    status, p = "excluded: atoms other than H, C, N", None
                else:
                    p = profile_atoms(at, f"{s['formula']}_{uid}")
                    if s["verdict"] != "ok":
                        status = f"excluded: screen {s['verdict']} (listed multiplicity {s['mult']}, bound {s['mult_min']})"
                    elif bas != "6-31G*":
                        status = f"excluded: basis {bas}"
                    elif p["frags"] != 1 or p["sp3C"] or p["CH2"] or p["bareC"] or p["pent"]:
                        status = "excluded: class " + ("superH" if p["sp3C"] or p["CH2"] else "dehydro" if p["bareC"] else "pentagon" if p["pent"] else "fragments")
                    elif flat >= 0.05:
                        status = f"excluded: not planar (rms {flat:.2f} A from the plane)"
                    elif not p["nN"] and p["nH"] != p["solo"] + p["duo"] + p["trio"] + p["quartet"]:
                        status = "excluded: H not on aromatic C"
                    else:
                        status = "use"
                g = p or dict(nH=s["nH"], nN=s["nN"], solo="", duo="", trio="", quartet="")
                rows.append(dict(species=f"{s['formula'].rstrip('+')}_v4u{uid}_q1", source="pahdb", uid=uid, structure_id=f"q1_{uid}",
                                 formula=s["formula"], charge=1, mult=s["mult"], nC=s["nC"], nH=g["nH"], nN=g["nN"],
                                 cls=("npah" if int(g["nN"] or 0) else "benzenoid") if p else "", solo=g["solo"], duo=g["duo"], trio=g["trio"],
                                 quartet=g["quartet"], lib300=0, lib600=0, lib900=0, status=status, basis=bas))
            el.clear()
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    from collections import Counter
    cat = rows[n0:]; c = Counter(re.sub(r" \(.*", "", r["status"]) for r in cat)
    print(f"{len(rows)} rows -> {out}: {n0} neutral rows ({sum(r['status'] == 'use' for r in rows[:n0])} used), {len(cat)} cations with 26-196 C")
    for k, v in c.most_common():
        print(f"  cations {v:5d}  {k}")
    use = [r for r in cat if r["status"] == "use"]; nc = sorted(int(r["nC"]) for r in use)
    print(f"cations used: {len(use)} (with N {sum(int(r['nN']) > 0 for r in use)}; multiplicity {dict(Counter(r['mult'] for r in use))}); "
          f"Nc {nc[0]}-{nc[-1]}, median {nc[len(nc) // 2]}")
    neu = [r for r in rows[:n0] if r["status"] == "use"]
    bins = [(20, 30), (30, 45), (45, 60), (60, 80), (80, 110), (110, 150), (150, 220)]
    for name, U in (("neutrals", neu), ("cations", use)):
        print(f"  {name} per size bin (species, with N, mean quartet share of H): " + ", ".join(
            f"{lo}-{hi}: {len(b)}, {sum(int(r['nN']) > 0 for r in b)}, {np.mean([int(r['quartet']) / max(int(r['nH']), 1) for r in b]) if b else 0:.2f}"
            for lo, hi in bins for b in [[r for r in U if lo <= int(r['nC']) < hi]]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "species_nc_v1.csv"))
