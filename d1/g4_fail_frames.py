# Frames along a G4 failure path, for DFT single points (is the model's bond breaking real, and AL data if not):
#   python g4_fail_frames.py <score.json> <md dir> <out frames dir> [--tags TAG,...] [--n-path 18] [--n-early 3]
# For each failing run of the g4_score.py output (or the --tags given): the bonds of its persistent change (lost and
# gained), and per saved frame s = max over those bonds of d / cutoff for a lost bond (>1: broken) and cutoff / d for
# a gained one. Picks frame 0 (the relaxed start), --n-early thermal frames from the first ps, and --n-path frames
# with s spread evenly from 0.8 to min(max s, 2.5) taken from the 3 ps before the persistent change began and the
# first 0.5 ps after it, where the change starts is the first break held for 100 fs (the approach to the break and the break itself, not the separated fragments).
# Writes <out>/<pair>_g4s<seed>T<T>_q<q>.extxyz in mlff_sample.py's format (graph_changed set per frame, so
# g16_io.py write stability-tests every changed frame).
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_integrity import FACTOR, RCOV, bonds


def parse_change(s):  # "+[H1-C11 C4-H20] -[C0-H1 ...]" -> gained, lost as index pairs
    g, l = re.match(r"\+\[(.*?)\] -\[(.*?)\]", s).groups()
    return [[tuple(int(re.sub(r"\D", "", x)) for x in b.split("-")) for b in part.split()] for part in (g, l)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("score"); ap.add_argument("md_dir"); ap.add_argument("out")
    ap.add_argument("--tags", default=""); ap.add_argument("--n-path", type=int, default=18); ap.add_argument("--n-early", type=int, default=3)
    a = ap.parse_args()
    from ase.io import Trajectory
    runs = json.load(open(a.score))["runs"]
    want = set(a.tags.split(",")) - {""}
    runs = [r for r in runs if (r["tag"] in want if want else not r["intact"])]
    os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(0)
    for r in runs:
        tr = Trajectory(f"{a.md_dir}/{r['tag']}.traj"); n = len(tr); sym = tr[0].get_chemical_symbols()
        gained, lost = parse_change(r["persistent_change"])
        cut = lambda i, j: FACTOR * (RCOV[sym[i]] + RCOV[sym[j]])
        P = np.array([tr[k].positions for k in range(n)]); b0 = bonds(sym, P[0])
        d = lambda k, i, j: np.linalg.norm(P[k, i] - P[k, j])
        s = np.array([max([d(k, i, j) / cut(i, j) for i, j in lost] + [cut(i, j) / d(k, i, j) for i, j in gained]) for k in range(n)])
        dt = 70.0 / (n - 1)                                  # ps per saved frame (20 ps NVT + 50 ps NVE)
        m = max(1, int(round(0.1 / dt)))                     # first break held for 100 fs (H hopping may come long before the final change)
        k1 = next((k for k in range(n - m) if (s[k:k + m] > 1).all()), int(round(r["final_change_onset_ps"] / dt))); lo = max(1, k1 - int(3 / dt)); hi = min(n - 1, k1 + int(0.5 / dt))
        win = np.arange(lo, hi + 1); smax = min(s[win].max(), 2.5)
        pick = [0] + sorted(rng.choice(np.arange(1, int(1 / dt)), a.n_early, replace=False).tolist())
        for t in np.linspace(0.8, smax, a.n_path):            # nearest unused window frame to each target s
            c = [k for k in win if k not in pick and s[k] <= 2.5]
            pick.append(min(c, key=lambda k: abs(s[k] - t)))
        name = f"{r['pair']}_g4s{r['seed']}T{r['temp']:.0f}_q{r['charge']}"
        with open(f"{a.out}/{name}.extxyz", "w") as f:
            for k in pick:
                ch = int(frozenset(bonds(sym, P[k])) != frozenset(b0))
                f.write(f"{len(sym)}\nProperties=species:S:1:pos:R:3 charge={r['charge']} mult={r['charge'] + 1} "
                        f"source={name}.xyz config_type=g4fail_{r['temp']:.0f}K sample_T={r['temp']:.0f} "
                        f"max_stretch={s[k]:.4f} max_elong=0.0000 graph_changed={ch} md_frame={k} md_ps={k * dt:.3f}\n")
                f.write("".join(f"{e} {x:.8f} {y:.8f} {z:.8f}\n" for e, (x, y, z) in zip(sym, P[k])))
        print(f"{name}: {len(pick)} frames, first held break {k1 * dt:.2f} ps, final change since {r['final_change_onset_ps']} ps, s range {s[pick].min():.2f}-{s[pick].max():.2f}, "
              f"changed {sum(frozenset(bonds(sym, P[k])) != frozenset(b0) for k in pick)}")


if __name__ == "__main__":
    main()
