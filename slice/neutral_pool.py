# Species pool for the neutral symbolic-regression prototype (coordinator request 2026-11-23).
#   python slice/neutral_pool.py [out csv, default slice/neutral_species_v1.csv]        (standard library only)
# Sources, all in the repo:
#   d1/new{300,600,900}_library_v1.csv        species with MLMD spectra at each temperature (the domain of the rules)
#   d1/new_species_v1.csv                     their structure profile (solo/duo/trio/quartet H, class)
#   d1/pahdb_v4_exclude_nonkekule_v1.csv      334 non-Kekule neutrals PAHdb lists as closed-shell singlets: removed
#   d1/pahdb_v4_replacements_v1.csv           200 of them replaced by our own high-spin spectra (d1/own_spectra/)
#   d1/pahdb_v4_screen_spin_v1.csv            d1/screen_spin.py verdicts (also covers PANHs and non-alternant species)
# A species is used when it is a planar benzenoid or N-containing PAH from the 300 K library, PAHdb lists it as a
# singlet, and the screen finds that consistent. Everything else is kept in the table with the reason in `status`,
# so that nothing is dropped silently. Our own high-spin species are added with source = own: they are not PAHdb
# data, and they have no MLMD spectra (the temperature rules are applied to them as to any harmonic line list).
# Charged species plug in as further rows: same structure_id, charge = +1.
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); D1 = os.path.join(HERE, "..", "d1")
sys.path.insert(0, D1)
from profile_xyz import profile_atoms  # noqa: E402

COLS = ["species", "source", "uid", "structure_id", "formula", "charge", "mult", "nC", "nH", "nN", "cls", "solo", "duo", "trio",
        "quartet", "lib300", "lib600", "lib900", "status"]


def rd(name):
    return list(csv.DictReader(open(os.path.join(D1, name))))


def own_geoms(path):
    out = {}
    with open(path) as f:
        while True:
            line = f.readline()
            if not line.strip():
                break
            n = int(line); name = f.readline().split()[0]
            out[name] = [(p[0], float(p[1]), float(p[2]), float(p[3])) for p in (f.readline().split() for _ in range(n))]
    return out


def main(out):
    lib = {T: {r["species"] for r in rd(f"new{T}_library_v1.csv")} for T in (300, 600, 900)}
    prof = {r["species"]: r for r in rd("new_species_v1.csv")}
    excl = {int(r["uid"]) for r in rd("pahdb_v4_exclude_nonkekule_v1.csv")}
    scr = {int(r["uid"]): r for r in rd("pahdb_v4_screen_spin_v1.csv")}
    rows = []
    for sp in sorted(lib[300] | lib[600] | lib[900]):
        p = prof[sp]; uid = int(p["uid"]); s = scr[uid]
        if uid in excl:
            status = "excluded: non-Kekule, listed as singlet (exclude list v1)"
        elif s["verdict"] != "ok":
            status = f"excluded: screen {s['verdict']} (listed multiplicity {s['mult']}, bound {s['mult_min']})"
        elif p["cls"] not in ("benzenoid", "npah"):
            status = f"excluded: class {p['cls']}"
        elif s["mult"] != "1":
            status = f"excluded: open-shell neutral (multiplicity {s["mult"]}); outside the closed-shell MLMD model"
        else:
            status = "use"
        rows.append(dict(species=sp, source="pahdb", uid=uid, structure_id=uid, formula=p["formula"], charge=0,
                         mult=s["mult"], nC=p["nC"], nH=p["nH"], nN=p["nN"], cls=p["cls"], solo=p["solo"], duo=p["duo"],
                         trio=p["trio"], quartet=p["quartet"], status=status,
                         **{f"lib{T}": int(sp in lib[T]) for T in (300, 600, 900)}))
    geo = own_geoms(os.path.join(D1, "own_spectra", "nonkekule_highspin_v1_geom.xyz"))
    for r in rd("own_spectra/nonkekule_highspin_v1_species.csv"):
        pr = profile_atoms(geo[r["species"]], r["species"])
        ok = int(r["n_imag"]) == 0 and pr["sp3C"] == 0 and pr["bareC"] == 0 and pr["pent"] == 0
        rows.append(dict(species=r["species"], source="own", uid=r["pahdb_v4_uid"], structure_id=r["pahdb_v4_uid"],
                         formula=r["formula"], charge=0, mult=r["mult"], nC=r["nC"], nH=r["nH"], nN=0, cls="benzenoid",
                         solo=pr["solo"], duo=pr["duo"], trio=pr["trio"], quartet=pr["quartet"], lib300=0, lib600=0, lib900=0,
                         status="use" if ok else "excluded: own spectrum fails checks"))
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    from collections import Counter
    c = Counter((r["source"], r["status"].split(" (")[0]) for r in rows)
    print(f"{len(rows)} rows -> {out}")
    for k in sorted(c):
        print(f"  {k[0]:6s} {c[k]:4d}  {k[1]}")
    use = [r for r in rows if r["status"] == "use"]
    nc = sorted(int(r["nC"]) for r in use)
    print(f"used: {len(use)} (PAHdb {sum(r['source'] == 'pahdb' for r in use)}, own {sum(r['source'] == 'own' for r in use)}; "
          f"with N {sum(int(r['nN']) > 0 for r in use)}); Nc {nc[0]}-{nc[-1]}, median {nc[len(nc) // 2]}")
    bins = [(20, 30), (30, 45), (45, 60), (60, 80), (80, 110), (110, 150), (150, 220)]
    print("  per size bin (species, share without solo H): " + ", ".join(
        f"{lo}-{hi}: {len(b)}, {sum(int(r['solo']) == 0 for r in b) / max(len(b), 1):.2f}"
        for lo, hi in bins for b in [[r for r in use if lo <= int(r['nC']) < hi]]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "neutral_species_v1.csv"))
