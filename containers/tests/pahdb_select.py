# Stage 2 (container B): keep hexagon-only, fully hydrogenated benzenoids; choose 20 per plan 4.1
# (five per 20-30, 30-50, 50-80, 80-130 C; named anchors first, then 3 compact + 2 irregular, spread in Nc).
# compact = no trio/quartet CH (plan 5.1 definition); irregular = has trio or quartet CH.
import json, sys, numpy as np, networkx as nx
from ase import Atoms
from ase.io import write
cand = json.load(open(sys.argv[1]))
ok = {}
for uid, s in cand.items():
    if s["n_h"] % 2 or not s["symmetry"].startswith("1-"): continue   # closed-shell singlets only (6C+H electrons: odd H = radical)
    g = np.array(s["geometry"]); Z = g[:, 0].astype(int); X = g[:, 1:]
    C = np.where(Z == 6)[0]; d = np.linalg.norm(X[C, None] - X[None, C], axis=-1)
    G = nx.Graph([(int(i), int(j)) for i, j in zip(*np.where(np.triu((d < 1.6) & (d > 0.5))))])
    if G.number_of_nodes() != len(C) or not nx.is_connected(G): continue
    rings = nx.minimum_cycle_basis(G)
    if not rings or any(len(r) != 6 for r in rings): continue
    if s["n_h"] != s["n_c"] - 2 * len(rings) + 2: continue       # fully hydrogenated, no holes
    if np.linalg.svd(X - X.mean(0), compute_uv=False)[-1] / np.sqrt(len(X)) > 0.1: continue   # planar (rms thickness < 0.1 A; molecules may be stored tilted)
    s["rings"] = len(rings); s["cls"] = "irregular" if (s["n_trio"] or s["n_quartet"]) else "compact"; ok[uid] = s
print(len(ok), "hexagon-only benzenoids;", sum(v["cls"] == "compact" for v in ok.values()), "compact")
ANCHORS = ["C24H12", "C32H14", "C54H18", "C96H24"]  # coronene, ovalene, circumcoronene, circumcircumcoronene (the "symmetry" field is the electronic state, so match by formula among compact species)
RANGES = [(20, 30), (30, 50), (50, 80), (80, 130)]
pick = []
for lo, hi in RANGES:
    inr = {u: s for u, s in ok.items() if lo <= s["n_c"] < hi}
    sel = [min((u for u, s in inr.items() if s["formula"] == f and s["cls"] == "compact"), key=int)
           for f in ANCHORS if any(s["formula"] == f and s["cls"] == "compact" for s in inr.values())]
    ncomp = sum(s["cls"] == "compact" for s in inr.values())
    want = {"compact": min(3, ncomp)}; want["irregular"] = 5 - want["compact"]   # small compact PAHs are scarce: top up with irregular
    for cls, n in want.items():
        pool = [u for u, s in inr.items() if s["cls"] == cls]
        have = sum(inr[u]["cls"] == cls for u in sel)
        for t in np.linspace(lo, hi - 1, max(n - have, 0) + 2)[1:-1]:        # evenly spaced Nc targets inside the range
            u = min((u for u in pool if u not in sel), key=lambda u: (abs(inr[u]["n_c"] - t), int(u))); sel.append(u)
    pick += [(lo, hi, u) for u in sel]
rows = ["idx,uid,formula,n_atoms,n_C,range,class,symmetry,n_solo,n_duo,n_trio,n_quartet"]; frames = []
prng = np.random.default_rng(7)
for i, (lo, hi, u) in enumerate(pick):
    s = ok[u]; g = np.array(s["geometry"])
    m = Atoms(numbers=g[:, 0].astype(int), positions=g[:, 1:] - g[:, 1:].mean(0))
    rows.append(f"{i},{u},{s['formula']},{len(m)},{s['n_c']},{lo}-{hi},{s['cls']},{s['symmetry']},{s['n_solo']},{s['n_duo']},{s['n_trio']},{s['n_quartet']}")
    for k in range(6):   # DFT geometry, then 5 copies with 0.05 A Gaussian displacements (plan 4.2)
        a = m.copy(); a.positions += (0 if k == 0 else 1) * prng.normal(0, 0.05, a.positions.shape)
        a.info.update(species=i, geom=k, uid=int(u)); frames.append(a)
write("/t/eq_frames.xyz", frames, format="extxyz"); open("/t/eq_species.csv", "w").write("\n".join(rows) + "\n")
print("\n".join(rows)); print(len(frames), "frames")
