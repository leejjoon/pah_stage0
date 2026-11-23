# Replica-to-replica scatter of band strengths in the D1 MLMD libraries (300/600/900 K, 4 NVE replicas each).
# Checks whether per-mode energy stays frozen in NVE (cloud-aimd finding, 2026-11-23): CV = std/mean of the band
# integral over the 4 replicas, per species, for the species present in all three libraries.
# Usage (from the repo root): python3 d1/replica_cv.py [spectra dir]
#   spectra dir: holds new300_v1, new300sp3_v1, new600_v1, new900_v1 (default /appsdata/$USER/d1/spectra)
import glob, os, sys, numpy as np, csv
S = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars("/appsdata/$USER/d1/spectra")
BANDS = {"3.3 CHstr 2950-3150": (2950, 3150), "6.2 1560-1660": (1560, 1660), "7.7 1250-1450": (1250, 1450),
         "8.6 1100-1250": (1100, 1250), "11.2 860-920": (860, 920), "oop 650-1000": (650, 1000),
         "total 500-3200": (500, 3200)}
lib = {T: {r[0] for r in csv.reader(open(f"d1/new{T}_library_v1.csv")) if r[0] != "species"} for T in (300, 600, 900)}
common = lib[300] & lib[600] & lib[900]
rows = []
for T, dirs in ((300, ["new300_v1", "new300sp3_v1"]), (600, ["new600_v1"]), (900, ["new900_v1"])):
    for d in dirs:
        for p in glob.glob(f"{S}/{d}/*_{T}K.npz"):
            sp = os.path.basename(p)[:-len(f"_{T}K.npz")]
            if sp not in common: continue
            z = np.load(p); f = z["freq"]; s = z["sigma"]; df = np.gradient(f)
            if s.shape[0] != 4: continue
            nC = int(sp[1:sp.index("H")])
            for b, (lo, hi) in BANDS.items():
                m = (f >= lo) & (f <= hi); st = (s[:, m] * df[m]).sum(1)
                rows.append((sp, nC, T, b, st.std(ddof=1) / st.mean() if st.mean() > 0 else np.nan))
import collections
print(f"species in all three libraries: {len(common)}; rows {len(rows)}")
by = collections.defaultdict(list)
for sp, nC, T, b, cv in rows: by[(b, T)].append(cv)
print(f"{'band':22s} " + "  ".join(f"{T}K med/p90" for T in (300, 600, 900)))
for b in BANDS:
    print(f"{b:22s} " + "  ".join(f"{np.nanmedian(by[(b,T)]):.3f}/{np.nanpercentile(by[(b,T)],90):.3f}" for T in (300, 600, 900)))
print("\nmedian CV by size bin (6.2 / 11.2 / total), 300 | 600 | 900 K")
for lo, hi in ((0, 30), (30, 50), (50, 70), (70, 200)):
    out = []
    for T in (300, 600, 900):
        out.append(" ".join(f"{np.nanmedian([cv for sp,nC,TT,bb,cv in rows if TT==T and bb==bb2 and lo<=nC<hi]):.3f}" for bb2 in ("6.2 1560-1660","11.2 860-920","total 500-3200")))
    n = len({sp for sp,nC,TT,bb,cv in rows if lo<=nC<hi})
    print(f"nC {lo:3d}-{hi:3d} (n={n:3d}): " + " | ".join(out))
