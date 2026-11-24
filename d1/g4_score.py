# G4 scoring (docs/plan/charged_mlff_proposal.md §5.7: bond graph intact after 20 ps NVT + 50 ps NVE, >= 99 % at
# 50-900 K, >= 95 % at 1200 K) of `mlff_accept.py md --traj-every N` runs, from their .traj and .json files:
#   python g4_score.py <md dir> <out.json> [--persist-fs 100] [--procs 16]
# Per run: the bond graph (mlmd_integrity.bonds, same cutoffs as the md check) of every saved frame against frame 0
# (the relaxed start). A change counts as persistent when it holds in every frame of the last
# --persist-fs, per bond: a bond lost (gained) in every frame of the window. A C-H stretched over the cutoff for an
# instant (common at 1200 K) does not count; one broken for good does, whatever other bonds flicker meanwhile.
# Also: time of the first change, onset of the final persistent change, time T first exceeds 3x the target (blow-up),
# NVE drift from the .json. Summary per temperature and charge: intact (persistent) / final-frame intact / runs.
import argparse, collections, glob, json, os, sys
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_integrity import bonds, describe


def score(args):
    path, persist_fs = args
    from ase.io import Trajectory
    r = json.load(open(path)); tag = os.path.basename(path)[:-5]
    tr = Trajectory(path[:-5] + ".traj"); n = len(tr)
    nvt_steps = int(r["nvt_ps"] * 2000); total = nvt_steps + int(r["nve_ps"] * 2000)
    every = total // (n - 2) if n > 2 else 1            # frames: NVT 0..nvt_steps and NVE 0..nve_steps, both inclusive
    fs = lambda k: (k * every if k * every <= nvt_steps else k * every - every) * 0.5
    sym = tr[0].get_chemical_symbols(); b0 = bonds(sym, tr[0].positions)
    graphs = [frozenset(bonds(sym, tr[k].positions)) for k in range(n)]
    changed = [g != b0 for g in graphs]
    m = max(1, int(round(persist_fs / (every * 0.5))))
    tail = graphs[-m:]                                   # per bond: lost (gained) in every frame of the window
    lost = set.intersection(*(set(b0 - g) for g in tail)); gained = set.intersection(*(set(g - b0) for g in tail))
    persistent = bool(lost or gained)
    onset = None
    if persistent:                                       # start of the last stretch over which every persistent change holds
        k = n - 1
        while k > 0 and lost <= set(b0 - graphs[k - 1]) and gained <= set(graphs[k - 1] - b0):
            k -= 1
        onset = fs(k) / 1000
    first = next((fs(k) / 1000 for k in range(n) if changed[k]), None)
    blow = next((step * 0.5 / 1000 + (r["nvt_ps"] if ph == "nve" else 0) for ph, step, T, _ in r["log"]
                 if T > 3 * r["temp"]), None)
    return dict(tag=tag, pair=r["pair"], charge=r["charge"], temp=r["temp"], seed=r["seed"], frames=n,
                final_intact=not changed[-1], intact=not persistent, persistent_change=describe(sym, gained, lost) if persistent else "",
                final_change=r["change"], frac_frames_changed=float(np.mean(changed)), first_change_ps=first,
                final_change_onset_ps=onset, blowup_ps=blow, nve_ps=r["nve_ps"], nve_drift_meV_atom=r["nve_drift_meV_atom"],
                nve_T_mean=r["nve_T_mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md_dir"); ap.add_argument("out")
    ap.add_argument("--persist-fs", type=float, default=100.0); ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    js = sorted(p for p in glob.glob(f"{a.md_dir}/*.json") if os.path.exists(p[:-5] + ".traj"))
    with Pool(a.procs) as pool:
        rows = pool.map(score, [(p, a.persist_fs) for p in js])
    S = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        for k in ((r["temp"], r["charge"]), (r["temp"], "all")):
            S[k][0] += r["intact"]; S[k][1] += r["final_intact"]; S[k][2] += 1
    print(f"### G4: intact (no persistent change over the last {a.persist_fs:.0f} fs) / final-frame intact / runs")
    for (T, q), (i, f, n) in sorted(S.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        lim = 95 if T >= 1200 else 99
        print(f"{T:6.0f} K q{q!s:3s} {i:3d} / {f:3d} / {n:3d}  {100 * i / n:5.1f} %  (G4 >= {lim} %){'  FAIL' if 100 * i / n < lim else ''}")
    print("\n### runs with any change in the final frame")
    for r in rows:
        if not r["final_intact"] or r["blowup_ps"] is not None:
            print(f"{r['tag']}: {'persistent' if not r['intact'] else 'transient'}; first change {r['first_change_ps']} ps, "
                  f"final change since {r['final_change_onset_ps']} ps, blow-up {r['blowup_ps']} ps, "
                  f"{100 * r['frac_frames_changed']:.1f} % frames changed, drift {r['nve_drift_meV_atom']:.2f} meV/atom; "
                  f"{(r['persistent_change'] or r['final_change'])[:160]}")
    d = [abs(r["nve_drift_meV_atom"]) for r in rows if r["blowup_ps"] is None]
    print(f"\n|NVE drift| (runs without blow-up): median {np.median(d):.2f}, max {max(d):.2f} meV/atom over {rows[0]['nve_ps']} ps")
    json.dump(dict(persist_fs=a.persist_fs, runs=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
