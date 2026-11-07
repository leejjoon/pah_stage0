# Stage 1 (container C): read PAHdb v4.00 and dump neutral, pure C/H, closed-shell, 20-130 C candidates with geometry.
# Run with --env TMPDIR=<local disk>: the suite caches a 267 MB pickle in tempfile.gettempdir().
import json, sys
from amespahdbpythonsuite.amespahdb import AmesPAHdb
db = AmesPAHdb(filename=sys.argv[1], check=False, cache=True)
ref = db.getdatabaseref(); print("PAHdb", ref["version"], ref["date"], len(ref["species"]), "species")
out = {}
for uid, s in ref["species"].items():
    if s["charge"] != 0 or not (20 <= s["n_c"] < 130): continue
    if any(s[k] for k in ("n_n", "n_o", "n_mg", "n_si", "n_fe", "n_ch2", "n_ch3")): continue
    if s["n_h"] % 2: continue                                  # 6C+H electrons: odd H is a radical; MACE-OFF is closed-shell only
    if s["formula"] != f"C{s['n_c']}H{s['n_h']}": continue
    out[int(uid)] = dict(formula=s["formula"], n_c=s["n_c"], n_h=s["n_h"], symmetry=s["symmetry"],
                         n_solo=s["n_solo"], n_duo=s["n_duo"], n_trio=s["n_trio"], n_quartet=s["n_quartet"],
                         geometry=[(int(a["type"]), a["x"], a["y"], a["z"]) for a in s["geometry"]])
json.dump(out, open(sys.argv[2], "w")); print(len(out), "candidates")
