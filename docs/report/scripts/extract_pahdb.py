"""Extract the PAHdb data the report's figures use (run in container C, CPU only):
  - stick spectra (frequency cm^-1, intensity km/mol) and 6 eV cascade emission spectra, Gaussian 15 cm^-1 FWHM,
    no redshift (plan 5.3), for four neutral/cation pairs;
  - 2-D geometries of the 25 neutral T0b species (slice/species.csv) for the molecule viewer.
Usage (from the repo root; keep it off GPU 0 / NUMA 0 while a timing run is going):
  numactl --cpunodebind=1 --membind=1 apptainer exec --containall --env TMPDIR=/cache \
    --bind <cache>:/cache --bind <data>:/data --bind $PWD:/r C_cpu.sif \
    python /r/docs/report/scripts/extract_pahdb.py /data/pahdb-complete-theoretical-v4.00.xml /r/docs/report/data/pahdb.json
"""
import sys, io, csv, json, contextlib
import numpy as np

xml, out = sys.argv[1], sys.argv[2]
with contextlib.redirect_stdout(io.StringIO()):
    from amespahdbpythonsuite.amespahdb import AmesPAHdb
    db = AmesPAHdb(filename=xml, check=False, cache=True)
ref = db.getdatabaseref()

species = list(csv.DictReader(open("/r/slice/species.csv")))
PAIRS = {"coronene": (18, 19), "circumcoronene": (37, 38), "C26H16": (814, 815), "circumcircumcoronene": (108, 111)}
uids = sorted({u for p in PAIRS.values() for u in p})

with contextlib.redirect_stdout(io.StringIO()):
    tr = db.gettransitionsbyuid(uids)
    sticks = {u: [[float(t["frequency"]), float(t["intensity"])] for t in tr.data[u]] for u in uids}
    tr.cascade(6 * 1.602176634e-12, multiprocessing=False, cache=False)
    grid = np.linspace(500.0, 4000.0, 3501)          # cm^-1, 1 cm^-1 step
    sp = tr.convolve(fwhm=15.0, gaussian=True, grid=grid, multiprocessing=False)
emis = {u: np.asarray(sp.data[u], dtype=float) for u in uids}

neutral = [s for s in species if s["charge"] == "0"]
with contextlib.redirect_stdout(io.StringIO()):
    geo = db.getgeometrybyuid([int(s["uid"]) for s in neutral])
mols = []
for s in neutral:
    g = geo.data[int(s["uid"])]
    X = np.array([[a["x"], a["y"], a["z"]] for a in g]); Z = [int(a["type"]) for a in g]
    X -= X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)  # project onto the molecular plane (PAHs are flat)
    P = X @ Vt[:2].T
    mols.append({k: s[k] for k in ("uid", "formula", "n_c", "n_h", "class", "n_solo", "n_duo", "n_trio", "n_quartet")} |
                {"Z": Z, "xy": np.round(P, 3).tolist(), "planarity_A": round(float(np.abs(X @ Vt[2]).max()), 3)})

info = {u: {"formula": ref["species"][u]["formula"], "charge": ref["species"][u]["charge"]} for u in uids}
json.dump({"pahdb": {"version": ref["version"], "date": str(ref["date"])}, "pairs": PAIRS, "info": info,
           "sticks": sticks, "grid_cm": grid.tolist(),
           "emission": {u: (e / e.max()).round(5).tolist() for u, e in emis.items()},
           "emission_max": {u: float(e.max()) for u, e in emis.items()},
           "molecules": mols}, open(out, "w"))
print(f"PAHdb {ref['version']}: {len(uids)} spectra, {len(mols)} geometries -> {out}")
