# Build 20 idealised benzenoid PAHs (hex lattice, C-C 1.40, C-H 1.09) across plan 4.1's ranges,
# plus 5 Gaussian-perturbed copies (0.05 A) each -> /t/eq_frames.xyz and /t/eq_species.csv.
# Stand-in for PAHdb geometries until the database is downloaded.
import numpy as np, networkx as nx
from ase import Atoms
from ase.io import write
A = 1.40
NB = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
def carbons(rings):
    c = []
    for q, r in rings:
        cx, cy = A*np.sqrt(3)*(q + r/2), A*1.5*r
        c += [(cx + A*np.cos(np.pi/6 + k*np.pi/3), cy + A*np.sin(np.pi/6 + k*np.pi/3)) for k in range(6)]
    return np.unique(np.round(np.array(c), 3), axis=0)
def build(rings):
    c = carbons(rings); d = np.linalg.norm(c[:, None] - c[None], axis=-1); nb = (d < 1.5) & (d > 0.1)
    hi = np.where(nb.sum(1) == 2)[0]
    h = [c[i] + 1.09*(2*c[i] - c[nb[i]].sum(0))/np.linalg.norm(2*c[i] - c[nb[i]].sum(0)) for i in hi]
    g = nx.Graph([(i, j) for i, j in zip(*np.where(np.triu(nb)))])
    kekule = len(nx.max_weight_matching(g, maxcardinality=True)) * 2 == len(c)
    holefree = len(h) == len(c) - 2*len(rings) + 2
    hh = np.linalg.norm(np.array(h)[:, None] - np.array(h)[None], axis=-1) + 9*np.eye(len(h))
    nofjord = hh.min() > 1.6          # coves/fjords put two H 0.5 A apart in a planar geometry; bays (1.71 A) are fine
    xyz = np.vstack([c, h]); xyz = np.c_[xyz, np.zeros(len(xyz))]
    return Atoms(f"C{len(c)}H{len(h)}", positions=xyz - xyz.mean(0)), kekule and holefree and nofjord
def grow(n_rings, compact, rng):
    rings = {(0, 0)}
    while len(rings) < n_rings:
        front = {(q+dq, r+dr) for q, r in rings for dq, dr in NB} - rings
        score = {f: sum((f[0]+dq, f[1]+dr) in rings for dq, dr in NB) for f in front}
        if compact:
            best = max(score.values()); cand = sorted(f for f in front if score[f] == best)
        else:   # irregular: random edge growth, skipping rings that would make a cove/fjord
            cand = [f for f in sorted(front) if score[f] <= 2]
            cand = [cand[i] for i in rng.permutation(len(cand))]
            cand = [f for f in cand if build(sorted(rings | {f}))[1]][:1]
            if not cand: return sorted(rings)
        rings.add(cand[rng.integers(len(cand))])
    return sorted(rings)
rng = np.random.default_rng(2026)
RANGES = [(20, 30), (30, 50), (50, 80), (80, 130)]
species = []
for lo, hi in RANGES:
    got, tries = [], 0
    while len(got) < 5 and tries < 5000:
        tries += 1
        compact = len(got) < 3                     # ~3 compact + 2 irregular per range
        n = int(rng.integers(3, 40)); rings = grow(n, compact, rng)
        m, ok = build(rings); nc = m.get_chemical_symbols().count("C")
        if ok and lo <= nc < hi and m.get_chemical_formula() not in [s[1].get_chemical_formula() for s in got]:
            got.append(("compact" if compact else "irregular", m))
    species += [(lo, hi, cls, m) for cls, m in got]
frames, rows = [], ["idx,formula,n_atoms,n_C,range,class"]
prng = np.random.default_rng(7)
for i, (lo, hi, cls, m) in enumerate(species):
    rows.append(f"{i},{m.get_chemical_formula()},{len(m)},{m.get_chemical_symbols().count('C')},{lo}-{hi},{cls}")
    for k in range(6):
        a = m.copy(); a.positions += (0 if k == 0 else 1) * prng.normal(0, 0.05, a.positions.shape)
        a.info.update(species=i, geom=k); frames.append(a)
write("/t/eq_frames.xyz", frames, format="extxyz")
open("/t/eq_species.csv", "w").write("\n".join(rows) + "\n"); print("\n".join(rows)); print(len(frames), "frames")
