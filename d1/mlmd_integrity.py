# Chemical integrity of mlmd_run.py trajectories: compare each system's bond graph in its last saved state
# (state.npz, end of the run or last checkpoint) with its input geometry. Catches what the run's QA flags miss:
# H loss or H hops, ring opening or closing, H2 formation. Bonds: distance < 1.2 x sum of covalent radii.
#   python mlmd_integrity.py <run outdir> <xyz dir>
#   python mlmd_integrity.py <xtb runs dir> -       (d1/xtb_md.py outputs: final positions vs the xyz they name)
# Prints one line per changed system and a summary per temperature. Positions only, no JAX.
import glob, json, os, sys
from collections import defaultdict
import numpy as np

RCOV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
FACTOR = 1.2


def read_xyz(path):
    with open(path) as f:
        n = int(f.readline()); f.readline()
        rows = [f.readline().split() for _ in range(n)]
    return [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])


def bonds(sym, p):
    r = np.array([RCOV[s] for s in sym])
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    b = (d < FACTOR * (r[:, None] + r[None])) & ~np.eye(len(sym), dtype=bool)
    return {(i, j) for i, j in zip(*np.nonzero(np.triu(b)))}


def describe(sym, gained, lost):
    tag = lambda ij: f"{sym[ij[0]]}{ij[0]}-{sym[ij[1]]}{ij[1]}"
    return f"+[{' '.join(map(tag, sorted(gained)))}] -[{' '.join(map(tag, sorted(lost)))}]"


def main():
    run, xyz = sys.argv[1], sys.argv[2]
    stats = defaultdict(lambda: [0, 0])                 # T -> [systems checked, changed]
    changed = []
    for f in sorted(glob.glob(f"{run}/*.npz")):          # xtb_md.py outputs: final positions + input xyz path
        z = np.load(f)
        if "positions" not in z.files:
            continue
        sym, p0 = read_xyz(str(z["xyz"])); T = float(z["T"])
        stats[T][0] += 1
        if bonds(sym, p0) != bonds(sym, z["positions"]):
            stats[T][1] += 1
            changed.append((T, os.path.basename(f)[:-4], 0, int(z["frames_done"]) == len(z["dipole"]),
                            describe(sym, bonds(sym, z["positions"]) - bonds(sym, p0), bonds(sym, p0) - bonds(sym, z["positions"]))))
    for meta in sorted(glob.glob(f"{run}/units/*/meta.json")):
        d = os.path.dirname(meta)
        if not os.path.exists(f"{d}/state.npz"):
            continue
        m = json.load(open(meta)); st = np.load(f"{d}/state.npz"); p = st["p"]
        done = os.path.exists(f"{d}/DONE")
        for k, s in enumerate(m["systems"]):
            sym, p0 = read_xyz(f"{xyz}/{s['species']}.xyz")
            b0, b1 = bonds(sym, p0), bonds(sym, p[k]) if np.isfinite(p[k]).all() else set()
            stats[s["T"]][0] += 1
            if b0 != b1:
                stats[s["T"]][1] += 1
                changed.append((s["T"], s["species"], s["replica"], done, describe(sym, b1 - b0, b0 - b1)))
    for T, sp, r, done, what in sorted(changed):
        print(f"{T:6.0f} K  {sp:16s} r{r} {'' if done else '(not finished) '}{what}")
    for T in sorted(stats):
        n, c = stats[T]
        print(f"T = {T:6.0f} K: {c}/{n} systems with a changed bond graph")


if __name__ == "__main__":
    main()
