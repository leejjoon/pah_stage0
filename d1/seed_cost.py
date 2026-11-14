# Seed-set DFT cost on one GPU from measured single-point times (G3 of docs/plan/handoff_gpu4pyscf_gh200.md).
# Interpolates t(nao) per charge (log-log, piecewise linear) on the C10-C144 series (pyscf_match.py JSONs, energy + forces + dipole), then sums over
# every seed frame (species_v2.csv pairs x frames per file). Optional: --pilot <dft_seed_run.py out dir> scales the
# fit by the measured/predicted ratio of the pilot frames (distorted geometries need more SCF cycles) and adds the
# measured stability-test cost per tested frame.
#   python seed_cost.py <timing dir> [--pilot DIR] [--speedup S]   (S: throughput gain per GPU from batching/MPS)
import argparse, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g16_io import read_extxyz

SEED = f"/var/tmp/{os.environ.get('USER')}-pah/d1/mlff_seed"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("timing"); ap.add_argument("--pilot")
    ap.add_argument("--speedup", type=float, default=1.0)
    a = ap.parse_args()
    fit = {}
    for q in (0, 1):
        pts = []
        for f in glob.glob(f"{a.timing}/*_q{q}_99x590_t9g6_gpu.json"):
            r = json.load(open(f))
            pts.append((r["tag"], r["nao"], r["t_total"]))
        # size variable: nao = 15 nC + 2 nH (6-31G*, 6D d), which also counts the H
        pts.sort(key=lambda p: p[1])
        x = np.log([p[1] for p in pts]); y = np.log([p[2] for p in pts])
        fit[q] = (lambda n, x=x, y=y: float(np.exp(np.interp(np.log(n), x, y))))   # seed nao lie inside the series
        print(f"q={q}: " + ", ".join(f"{t} nao={n} {s:.0f}s" for t, n, s in pts))
    tot = {0: 0.0, 1: 0.0}; nfr = 0; frames = {}
    for f in sorted(glob.glob(f"{SEED}/frames/*.extxyz")) + sorted(glob.glob(f"{SEED}/frames_transfer/*.extxyz")):
        tag = os.path.basename(f)[:-7]; q = int(tag[-1])
        fr = read_extxyz(f); sym = fr[0][1]
        nao = 15 * sym.count("C") + 2 * sym.count("H")
        tot[q] += len(fr) * fit[q](nao); nfr += len(fr)
        for k in range(len(fr)):
            frames[f"{tag}_f{k:03d}"] = (q, nao)
    pred = tot[0] + tot[1]
    print(f"{nfr} frames; predicted from the series: q0 {tot[0] / 3600:.1f} h + q1 {tot[1] / 3600:.1f} h = {pred / 3600:.1f} GPU-h")
    if a.pilot:
        R = [json.load(open(f)) for f in glob.glob(f"{a.pilot}/*.json")]
        meas = sum(r["t_wall"] - r["t_stab"] for r in R)
        p = sum(fit[frames[r["name"]][0]](frames[r["name"]][1]) for r in R)
        ratio = meas / p
        st = [r for r in R if r["stability_test"]]
        # stability cost relative to the frame's predicted single point
        st_ratio = (np.mean([r["t_stab"] / fit[frames[r["name"]][0]](frames[r["name"]][1])
                             for r in st]) if st else 0.0)
        man = {}
        for m in (f"{SEED}/gjf/manifest.json", f"{SEED}/gjf_transfer/manifest.json"):
            man.update(json.load(open(m)))
        st_pred = sum(fit[q](nao) for n, (q, nao) in frames.items() if man.get(n, {}).get("stability_test"))
        total = pred * ratio + st_pred * st_ratio
        print(f"pilot: {len(R)} frames, measured/predicted single-point time {ratio:.2f}; stability tests {len(st)}, "
              f"cost {st_ratio:.1f}x a predicted single point")
        print(f"seed set: {pred * ratio / 3600:.1f} GPU-h single points + {st_pred * st_ratio / 3600:.1f} GPU-h stability "
              f"= {total / 3600:.1f} GPU-h on one H100; {total / 3600 / a.speedup:.1f} h at {a.speedup}x per GPU; "
              f"{total / 3600 / a.speedup / 2:.1f} h on both GPUs")


if __name__ == "__main__":
    main()
