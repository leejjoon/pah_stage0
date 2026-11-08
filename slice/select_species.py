# T0b species selection (plan 5.1), run in container C:
#   apptainer exec --containall --env TMPDIR=/cache --bind <cache>:/cache --bind <data>:/data --bind <repo>:/r C_cpu.sif \
#     python /r/slice/select_species.py /data/pahdb-complete-theoretical-v4.00.xml /r/slice
# 25 molecules in both neutral and cation states (50 spectra), size ladder 20-30, 30-45, 45-60, 60-80, 80-110 C,
# about 18 compact + 7 irregular. Pure C/H, hexagon-only, fully hydrogenated, no aliphatic groups.
# Neutral/cation pairs are selected first (cations are scarce at the small end).
# Solo/duo/trio/quartet H counts come from the suite's boundary-edge code (Geometry.bec(); Ricca+2025 2.4, 3.2)
# and are cross-checked against (a) PAHdb's stored n_solo... fields and (b) an independent count from the geometry.
import sys, io, contextlib, collections, json
import numpy as np

xml, outdir = sys.argv[1], sys.argv[2]
with contextlib.redirect_stdout(io.StringIO()):
    from amespahdbpythonsuite.amespahdb import AmesPAHdb
    db = AmesPAHdb(filename=xml, check=False, cache=True)
dbref = db.getdatabaseref()
ref = dbref["species"]
print(f"PAHdb {dbref['version']} ({dbref['date']}), {len(ref)} species")

BINS = [(20, 30), (30, 45), (45, 60), (60, 80), (80, 110)]
QUOTA = {"compact": [4, 4, 4, 3, 3], "irregular": [1, 1, 1, 2, 2]}     # 18 + 7 = 25 molecules
ANCHORS = ["C24H12", "C32H14", "C54H18", "C96H24"]                      # coronene, ovalene, circumcoronene, circumcircumcoronene
def binof(nc):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= nc < hi or (i == len(BINS) - 1 and nc == hi): return i
    return None

# 1. basic filters on the stored fields
base = {u: s for u, s in ref.items()
        if s["charge"] in (0, 1) and binof(s["n_c"]) is not None and s["formula"].rstrip("+-") == f"C{s['n_c']}H{s['n_h']}"   # cation formulas carry a trailing "+"
        and not any(s[k] for k in ("n_n", "n_o", "n_mg", "n_si", "n_fe", "n_ch2", "n_ch3"))}
uids = sorted(base)
geo = db.getgeometrybyuid(uids)
rings, bec = geo.rings(), geo.bec()

def geometric_counts(s):
    """Independent solo/duo/trio/quartet H counts: CH carbons grouped into runs of bonded CH carbons."""
    g = s["geometry"]; Z = np.array([a["type"] for a in g]); X = np.array([[a["x"], a["y"], a["z"]] for a in g])
    C, H = np.where(Z == 6)[0], np.where(Z == 1)[0]
    dCH = np.linalg.norm(X[H, None] - X[None, C], axis=-1)
    ch = C[dCH.argmin(1)]                                   # carbon carrying each H
    ch = np.unique(ch)
    dCC = np.linalg.norm(X[ch, None] - X[None, ch], axis=-1)
    adj = (dCC < 1.6) & (dCC > 0.5)
    parent = list(range(len(ch)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in zip(*np.where(np.triu(adj))): parent[find(i)] = find(j)
    sizes = collections.Counter(collections.Counter(find(i) for i in range(len(ch))).values())
    return {k: sizes.get(n, 0) * n for n, k in ((1, "solo"), (2, "duo"), (3, "trio"), (4, "quartet"))}, \
           {n: c for n, c in sizes.items() if n > 4}

def bec_counts(code):
    c = collections.Counter(code)
    return {"solo": c["2"], "duo": 2 * c["3"], "trio": 3 * c["4"], "quartet": 4 * c["5"]}, c["1"], \
           sum(c[d] for d in c if d not in "12345")

def canon(code):
    """Rotation/reflection-invariant form of the cyclic boundary-edge code."""
    rots = [code[i:] + code[:i] for i in range(len(code))]; r = code[::-1]
    return min(rots + [r[i:] + r[:i] for i in range(len(r))])

def spectrum(s):
    """Sorted eigenvalues of the carbon adjacency matrix (graph invariant used to confirm neutral/cation pairing)."""
    g = s["geometry"]; X = np.array([[a["x"], a["y"], a["z"]] for a in g if a["type"] == 6])
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    return np.round(np.linalg.eigvalsh(((d < 1.6) & (d > 0.5)).astype(float)), 6)

ok, reject = {}, collections.Counter()
mism_fields, mism_geom = [], []
for u in uids:
    s = base[u]; r = rings[u]
    if r["six"] == 0 or any(r[k] for k in ("three", "four", "five", "seven", "eight")): reject["non-hexagon ring"] += 1; continue
    if s["n_h"] != s["n_c"] - 2 * r["six"] + 2: reject["not fully hydrogenated / holes"] += 1; continue
    if s["charge"] == 0 and (s["n_h"] % 2 or not s["symmetry"].startswith("1-")): reject["neutral not closed-shell"] += 1; continue
    code = bec[u]
    bc, nbay, other = bec_counts(code)
    if other or sum(bc.values()) != s["n_h"]: reject["boundary code inconsistent with n_h"] += 1; continue
    gc, big = geometric_counts(s)
    stored = {"solo": s["n_solo"], "duo": s["n_duo"], "trio": s["n_trio"], "quartet": s["n_quartet"]}
    if stored != bc: mism_fields.append(u)
    if gc != bc or big: mism_geom.append(u)
    cls = "compact" if (bc["trio"] == 0 and bc["quartet"] == 0) else "irregular"   # plan 5.1 definition; bays recorded as n_bay
    ok[u] = dict(uid=u, formula=s["formula"].rstrip("+-"), n_c=s["n_c"], n_h=s["n_h"], charge=s["charge"], symmetry=s["symmetry"],
                 bec=code, n_bay=nbay, cls=cls, **{f"n_{k}": v for k, v in bc.items()},
                 key=(s["formula"].rstrip("+-"), canon(code)))
print("filters:", dict(reject), f"-> {len(ok)} pure hexagon-only PAHs (neutral closed-shell or cation), 20-110 C")
print(f"bec counts vs stored n_solo/n_duo/n_trio/n_quartet: {len(ok) - len([u for u in mism_fields if u in ok])}/{len(ok)} agree;"
      f" vs geometric count: {len(ok) - len([u for u in mism_geom if u in ok])}/{len(ok)} agree")
for name, lst in (("stored fields", mism_fields), ("geometry", mism_geom)):
    for u in [u for u in lst if u in ok][:5]: print(f"  disagrees with {name}: UID {u} {ok[u]['formula']} bec {ok[u]['bec']}")
n2130 = collections.Counter(v["charge"] for v in ok.values() if 21 <= v["n_c"] <= 30)
print(f"21-30 C after filters: {n2130[0]} neutral, {n2130[1]} cation (Ricca+2025 Table 1, pure PAHs before our filters: 786 / 30)")

# 2. neutral/cation pairs: same formula and boundary code, confirmed by the carbon-graph spectrum
neu = collections.defaultdict(list); cat = collections.defaultdict(list)
for u, v in ok.items(): (neu if v["charge"] == 0 else cat)[v["key"]].append(u)
pairs = []
for k in neu:
    if k not in cat: continue
    un, uc = min(neu[k]), min(cat[k])
    if not np.allclose(spectrum(ref[un]), spectrum(ref[uc]), atol=1e-4): continue
    pairs.append((un, uc))
print(f"neutral/cation pairs: {len(pairs)}")
P = {un: (un, uc) for un, uc in pairs}
pool = collections.defaultdict(list)
for un, uc in pairs: pool[(binof(ok[un]["n_c"]), ok[un]["cls"])].append(un)
print("pairs per bin (compact/irregular):", [(f"{lo}-{hi}", len(pool[(i, 'compact')]), len(pool[(i, 'irregular')])) for i, (lo, hi) in enumerate(BINS)])

# 3. selection: anchors first, then evenly spaced Nc targets per bin and class; distinct formulas where possible
sel = []
for i, (lo, hi) in enumerate(BINS):
    short = 0
    for cls in ("compact", "irregular"):
        cand = sorted(pool[(i, cls)], key=int); n = QUOTA[cls][i] + (short if cls == "irregular" else 0)
        if len(cand) < n:
            print(f"  bin {lo}-{hi} {cls}: only {len(cand)} pairs for {n} slots" + ("; the rest go to irregular" if cls == "compact" else ""))
            short = n - len(cand)
        mine = [u for u in cand if cls == "compact" and ok[u]["formula"] in ANCHORS and ok[u]["n_c"] and u == min(
            (w for w in cand if ok[w]["formula"] == ok[u]["formula"]), key=int)][:n]
        for t in np.linspace(lo, hi, max(n - len(mine), 0) + 2)[1:-1]:
            left = [u for u in cand if u not in mine]
            if not left: break
            used = {ok[u]["formula"] for u in mine}
            u = min(left, key=lambda u: (ok[u]["formula"] in used, cls == "compact" and ok[u]["n_bay"] > 0, abs(ok[u]["n_c"] - t), int(u)))
            mine.append(u)
        sel += sorted(mine, key=lambda u: ok[u]["n_c"])

rows = []
for pid, un in enumerate(sel):
    for u in P[un]:
        v = ok[u]
        rows.append([pid, v["uid"], v["formula"], v["n_c"], v["n_h"], v["charge"], v["cls"], v["n_solo"], v["n_duo"], v["n_trio"],
                     v["n_quartet"], v["n_bay"], v["bec"], v["symmetry"], ref[u]["method"]])
hdr = "pair,uid,formula,n_c,n_h,charge,class,n_solo,n_duo,n_trio,n_quartet,n_bay,bec,symmetry,method"
# method is RB3LYP for every (closed-shell) neutral and UB3LYP for every (doublet) cation: it tracks charge, not size
assert {(ok[u]["charge"], ref[u]["method"]) for un in sel for u in P[un]} == {(0, "RB3LYP"), (1, "UB3LYP")}
open(f"{outdir}/species.csv", "w").write(hdr + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")
print(hdr); [print(",".join(map(str, r))) for r in rows]

# 4. checks the plan asks for
print("achieved:", {c: sum(ok[u]["cls"] == c for u in sel) for c in ("compact", "irregular")}, "(plan: about 18 compact / 7 irregular)")
print("method by charge:", sorted({(ok[u]["charge"], ref[u]["method"]) for un in sel for u in P[un]}))
print("== Nc range per compactness class (neutral and cation are the same molecules by construction):")
for cls in ("compact", "irregular"):
    for q in (0, 1):
        nc = [r[3] for r in rows if r[6] == cls and r[5] == q]
        print(f"  {cls:9s} {'neutral' if q == 0 else 'cation '}: n={len(nc)}, Nc {min(nc)}-{max(nc)}, median {np.median(nc):.0f}")
print("== fraction of molecules without solo H, per size bin:")
for i, (lo, hi) in enumerate(BINS):
    m = [r for r in rows if r[5] == 0 and binof(r[3]) == i]
    print(f"  {lo}-{hi} C: {sum(r[7] == 0 for r in m)}/{len(m)} without solo H")
json.dump({"pahdb_version": dbref["version"], "pahdb_date": dbref["date"], "bins": BINS, "quota": QUOTA,
           "compact_definition": "no trio and no quartet CH (plan 5.1); bay-free species preferred among compact; n_bay = count of '1' segments in the boundary-edge code",
           "pairs_available": len(pairs),
           "achieved": {c: sum(ok[u]["cls"] == c for u in sel) for c in ("compact", "irregular")},
           "note": "plan 5.1 asks for about 18 compact / 7 irregular; the 20-30 C bin has a single compact neutral/cation pair "
                   "(coronene), so its other slots are irregular, and 30-45 C needs a bay-bearing compact species (C44H18)"}, open(f"{outdir}/species_meta.json", "w"), indent=1)

# 5. drawing check: every selected neutral, H coloured by the independent geometric group count, bec counts in the title
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
col = {1: "tab:red", 2: "tab:blue", 3: "tab:green", 4: "tab:purple"}
fig, axs = plt.subplots(5, 5, figsize=(15, 15))
for ax, un in zip(axs.flat, sel):
    g = ref[un]["geometry"]; Z = np.array([a["type"] for a in g]); X = np.array([[a["x"], a["y"], a["z"]] for a in g])
    X = X - X.mean(0); X = X @ np.linalg.svd(X, full_matrices=False)[2].T          # rotate into the molecular plane
    C, H = np.where(Z == 6)[0], np.where(Z == 1)[0]
    d = np.linalg.norm(X[C, None] - X[None, C], axis=-1)
    for a, b in zip(*np.where(np.triu((d < 1.6) & (d > 0.5)))): ax.plot(*X[[C[a], C[b]], :2].T, "k-", lw=0.8)
    hc = C[np.linalg.norm(X[H, None] - X[None, C], axis=-1).argmin(1)]
    chs = np.unique(hc); dd = np.linalg.norm(X[chs, None] - X[None, chs], axis=-1); adj = (dd < 1.6) & (dd > 0.5)
    lab = -np.ones(len(chs), int)
    for i in range(len(chs)):
        if lab[i] < 0:
            stack, lab[i] = [i], i
            while stack:
                k = stack.pop()
                for j in np.where(adj[k] & (lab < 0))[0]: lab[j] = i; stack.append(j)
    size = {c: (lab == lab[i]).sum() for i, c in enumerate(chs)}
    for h, c in zip(H, hc): ax.plot(*X[h, :2], "o", color=col.get(size[c], "k"), ms=4)
    v = ok[un]
    ax.set_title(f"{v['formula']} UID {un}/{P[un][1]} {v['cls']}\nbec s{v['n_solo']} d{v['n_duo']} t{v['n_trio']} q{v['n_quartet']} bay{v['n_bay']}", fontsize=8)
    ax.set_aspect("equal"); ax.axis("off")
fig.suptitle("T0b species (neutral geometry). H colour from geometry: red solo, blue duo, green trio, purple quartet", fontsize=11)
fig.tight_layout(); fig.savefig(f"{outdir}/species_check.png", dpi=110)
print("wrote species.csv, species_meta.json, species_check.png")
