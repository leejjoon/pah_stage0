# Active-learning round for the charged MLFF (docs/plan/charged_mlff_plan.md, Phase 3): explore with one model,
# score the saved frames by committee force disagreement, pick the most uncertain frames per trajectory and package
# them for DFT exactly like the earlier extra-frame batches (frames_<tag>/ + gjf_<tag>/manifest.json, read by
# d1/dft_seed_run.py run; see docs/plan/hangang_hot_t2md_handoff.md).
#
#   python mlff_al_select.py jobs <species.csv> [--charges 1] [--temps 300,900,1200] [--seeds 0]  > jobs.txt
#       one line "<pair> <charge> <T> <seed>" per run, train-split pairs only.
#   python mlff_al_select.py explore <model> <species.csv> <xyz dir> <outdir> --pair P --charge q --temp T [--seed s]
#        [--ps 10] [--equil-ps 1] [--every 100] [--device cuda] [--round al1]
#       relax (fmax 1e-3), Maxwell-Boltzmann start, ASE Langevin NVT (0.5 fs, friction 0.02, fixcm), a frame every
#       --every steps (100 = 50 fs) after --equil-ps, COM-centred, as <outdir>/<pair>_q<q>_T<T>_s<s>.extxyz with
#       mlff_sample.py's labels (max_stretch, max_elong, graph_changed against the relaxed start) and changed_n (of
#       this frame, +1, +2 saved frames, how many show a changed graph: 3 = persistent). The run stops at the first
#       persistent change; frames up to it are kept (committee review B2: no censoring). Lock files let several
#       workers share one job list, as in mlff_accept.py md.
#   python mlff_al_select.py score <outdir> --models m1,m2,... [--device cuda]
#       for every explore file: forces of each model; per-atom disagreement s_i = sqrt(mean_m |F_m,i - <F>_i|^2)
#       (eV/A), frame score = max_i s_i (also the mean). Written to <file>.score.json; files already scored with the
#       same model list are skipped.
#   python mlff_al_select.py select <outdir> <species.csv> <xyz dir> <frames dir> [--lo 0.1] [--hi 2.0]
#        [--per-traj 10] [--gap-ps 0.5] [--budget 4000] [--stable-frac 0.3]
#       candidates: lo <= score <= hi (above hi: the model has left the physical surface; reported, not picked);
#       per trajectory greedy by score, at least --gap-ps apart, at most --per-traj; then the --budget highest overall.
#       Writes <frames dir>/<pair>_q<q>_<tag>.extxyz (tag = basename of frames dir minus "frames_", e.g. al1): the
#       DFT minimum from <xyz dir> first (the per-run energy reference of mlff_qa.py; already labelled, drop the
#       duplicate when building the dataset), then the picked frames; then g16_io.py write fills
#       ../gjf_<tag>/manifest.json (stability-test choice by its hash rule, --stable-frac of frames at >= 1200 K).
#       Prints the score distribution per charge and temperature and writes <frames dir>/selection.csv.
# Leakage guard: jobs, explore and select exit if a pair is not in the train split of <species.csv> (held-out
# species must never get training frames).
import argparse, csv, glob, json, os, sys, time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlff_sample import bond_list, centred, graph, labels, write_frame
from mlff_accept import MASS, atoms_from, calculator, relax
from g16_io import read_extxyz

MULT = {0: 1, 1: 2}


def train_pairs(species):
    rows = list(csv.DictReader(open(species)))
    return {r["pair"] for r in rows if r["split"] == "train"}


def require_train(species, pairs):
    bad = sorted(set(pairs) - train_pairs(species))
    if bad:
        sys.exit(f"not in the train split of {species}: {', '.join(bad)} (held-out pairs never get AL frames)")


def cmd_jobs(a):
    ok = train_pairs(a.species)
    excl = {x for x in a.exclude.split(",") if x}
    for p in sorted(ok - excl):
        for q in map(int, a.charges.split(",")):
            for T in a.temps.split(","):
                for s in a.seeds.split(","):
                    print(p, q, T, s)


def tag_of(pair, q, T, s):
    return f"{pair}_q{q}_T{int(float(T))}_s{s}"


def cmd_explore(a):
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    require_train(a.species, [a.pair])
    os.makedirs(a.outdir, exist_ok=True)
    tag = tag_of(a.pair, a.charge, a.temp, a.seed); out = f"{a.outdir}/{tag}.extxyz"
    if os.path.exists(out):
        return
    try:
        os.close(os.open(f"{a.outdir}/{tag}.lock", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        print(f"{tag}: locked by another worker (delete {tag}.lock if that worker died)", flush=True); return
    t0 = time.time()
    at = atoms_from(f"{a.xyz}/{a.pair}_q{a.charge}.xyz"); at.calc = calculator(a.model, a.charge, a.device, "float32")
    relax(at, 1e-3, 2000)
    sym = at.get_chemical_symbols(); m = np.array([MASS[s] for s in sym])
    p0 = at.positions.copy(); ref = {"bonds": bond_list(sym, p0), "graph": graph(sym, p0)}
    MaxwellBoltzmannDistribution(at, temperature_K=a.temp, rng=np.random.default_rng(a.seed)); Stationary(at)
    dyn = Langevin(at, 0.5 * units.fs, temperature_K=a.temp, friction=0.02, fixcm=True,
                   rng=np.random.default_rng(a.seed + 1))
    n_eq, n_tot = int(a.equil_ps * 2000), int(a.ps * 2000)
    dyn.run(n_eq)
    saved, stop = [], None                                  # (t_ps, positions, labels, E_driver)
    while dyn.nsteps < n_tot:
        dyn.run(a.every)
        p = at.positions.copy(); lab = labels(sym, p, ref)
        saved.append((dyn.nsteps * 0.5e-3, p, lab, float(at.get_potential_energy()), float(at.get_temperature())))
        if len(saved) >= 3 and all(s[2]["graph_changed"] for s in saved[-3:]):
            stop = saved[-3][0]; break                      # persistent change: stop, keep what we have
    q = a.charge
    with open(out + ".tmp", "w") as f:
        for k, (t, p, lab, e, T) in enumerate(saved):
            cn = sum(s[2]["graph_changed"] for s in saved[k:k + 3]) if k + 3 <= len(saved) else "none"
            write_frame(f, sym, centred(p, m), dict(
                charge=q, mult=MULT[q], source=f"{a.pair}_q{q}.xyz", config_type=f"{a.round}_md_{int(a.temp)}K",
                sample_T=f"{a.temp:g}", **lab, changed_n=cn, t_ps=f"{t:.3f}", traj=tag, al_seed=a.seed,
                driver_energy=f"{e:.6f}", inst_T=f"{T:.0f}"))
    os.replace(out + ".tmp", out); os.remove(f"{a.outdir}/{tag}.lock")
    print(f"{tag}: {len(saved)} frames, {'stopped at persistent change t=%.2f ps' % stop if stop else 'no persistent change'}"
          f", wall {time.time() - t0:.0f} s", flush=True)


def cmd_score(a):
    models = a.models.split(",")
    calcs = {}
    for path in sorted(glob.glob(f"{a.outdir}/*.extxyz")):
        sp = path + ".score.json"
        if os.path.exists(sp) and json.load(open(sp)).get("models") == models:
            continue
        frs = read_extxyz(path)
        if not frs:
            continue
        q = int(frs[0][0]["charge"])
        if q not in calcs:
            calcs[q] = [calculator(mp, q, a.device, "float32") for mp in models]
        from ase import Atoms
        smax, smean = [], []
        for info, sym, pos in frs:
            at = Atoms(sym, positions=pos)
            F = []
            for c in calcs[q]:
                at.calc = c; F.append(at.get_forces().copy())
            F = np.array(F); s = np.sqrt(((F - F.mean(0)) ** 2).sum(-1).mean(0))
            smax.append(float(s.max())); smean.append(float(s.mean()))
        json.dump(dict(models=models, score_max=smax, score_mean=smean), open(sp, "w"))
        print(f"{os.path.basename(path)}: {len(frs)} frames, max score {max(smax):.3f} eV/A", flush=True)


def cmd_select(a):
    tag = os.path.basename(os.path.normpath(a.frames_dir))
    if not tag.startswith("frames_"):
        sys.exit("frames dir must be named frames_<tag> (dft_seed_run.py finds ../gjf_<tag>/manifest.json from it)")
    tag = tag[7:]
    cand, dist, n_hi = defaultdict(list), defaultdict(list), defaultdict(int)
    pairs = set()
    for path in sorted(glob.glob(f"{a.outdir}/*.extxyz")):
        sp = path + ".score.json"
        if not os.path.exists(sp):
            sys.exit(f"{path}: not scored (run score first)")
        sc = json.load(open(sp))["score_max"]
        for (info, sym, pos), s in zip(read_extxyz(path), sc):
            info = {k: v.strip('"') for k, v in info.items() if k != "Properties"}
            pair = info["source"].rsplit("_q", 1)[0]; q = int(info["charge"]); pairs.add(pair)
            dist[(q, info["sample_T"])].append(s)
            if s > a.hi:
                n_hi[(q, info["sample_T"])] += 1
            elif s >= a.lo:
                cand[info["traj"]].append((s, float(info["t_ps"]), pair, q, info, sym, pos))
    require_train(a.species, pairs)
    picked = []
    for traj, c in cand.items():
        c.sort(key=lambda x: -x[0]); keep = []
        for x in c:
            if len(keep) < a.per_traj and all(abs(x[1] - y[1]) >= a.gap_ps for y in keep):
                keep.append(x)
        picked += keep
    picked.sort(key=lambda x: -x[0]); n_cand = len(picked); picked = picked[:a.budget]
    print(f"score (max over atoms of committee force std, eV/A) per charge and T; lo={a.lo} hi={a.hi}")
    print("| q | T | frames | median | p90 | p98 | max | > hi | picked |\n|---|---|---|---|---|---|---|---|---|")
    for (q, T), v in sorted(dist.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        v = np.array(v); npk = sum(1 for x in picked if x[3] == q and x[4]["sample_T"] == T)
        print(f"| {q} | {T} | {len(v)} | {np.median(v):.3f} | {np.quantile(v, .9):.3f} | {np.quantile(v, .98):.3f} "
              f"| {v.max():.3f} | {n_hi[(q, T)]} | {npk} |")
    print(f"{n_cand} candidates after per-trajectory selection, {len(picked)} kept (budget {a.budget})")
    os.makedirs(a.frames_dir, exist_ok=True)
    by_pq = defaultdict(list)
    for x in picked:
        by_pq[(x[2], x[3])].append(x)
    gjf = os.path.join(os.path.dirname(os.path.abspath(a.frames_dir)), f"gjf_{tag}")
    import argparse as _ap
    from g16_io import cmd_write
    rows = []
    for (pair, q), xs in sorted(by_pq.items()):
        xs.sort(key=lambda x: (float(x[4]["sample_T"]), x[4]["traj"], x[1]))
        fn = f"{a.frames_dir}/{pair}_q{q}_{tag}.extxyz"
        at = atoms_from(f"{a.xyz}/{pair}_q{q}.xyz"); sym = at.get_chemical_symbols(); p = at.positions
        ref = {"bonds": bond_list(sym, p), "graph": graph(sym, p)}
        m = np.array([MASS[s] for s in sym])
        with open(fn, "w") as f:
            write_frame(f, sym, centred(p, m), dict(charge=q, mult=MULT[q], source=f"{pair}_q{q}.xyz", config_type="min",
                                                     sample_T="none", **labels(sym, p, ref)))
            for k, (s, t, _, _, info, sy, pos) in enumerate(xs, 1):
                info = {**info, "al_score": f"{s:.4f}"}
                write_frame(f, sy, pos, info)
                rows.append(dict(frame=f"{pair}_q{q}_{tag}_f{k:03d}", pair=pair, charge=q, sample_T=info["sample_T"],
                                 traj=info["traj"], t_ps=info["t_ps"], score=f"{s:.4f}",
                                 graph_changed=info["graph_changed"], changed_n=info["changed_n"]))
        cmd_write(_ap.Namespace(extxyz=fn, outdir=gjf, tag=None, basis="6-31G*", nproc=8, mem="16GB",
                                stable_hot_T=1200.0, stable_frac=a.stable_frac, no_stable=False))
    with open(f"{a.frames_dir}/selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(by_pq)} files ({len(picked)} picked + {len(by_pq)} min frames) -> {a.frames_dir}, manifest {gjf}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs"); j.add_argument("species")
    j.add_argument("--charges", default="1"); j.add_argument("--temps", default="300,900,1200")
    j.add_argument("--seeds", default="0"); j.add_argument("--exclude", default="")
    e = sub.add_parser("explore")
    for k in ("model", "species", "xyz", "outdir"):
        e.add_argument(k)
    e.add_argument("--pair", required=True); e.add_argument("--charge", type=int, required=True)
    e.add_argument("--temp", type=float, required=True); e.add_argument("--seed", type=int, default=0)
    e.add_argument("--ps", type=float, default=10); e.add_argument("--equil-ps", type=float, default=1)
    e.add_argument("--every", type=int, default=100); e.add_argument("--device", default="cuda")
    e.add_argument("--round", default="al1")
    s = sub.add_parser("score"); s.add_argument("outdir"); s.add_argument("--models", required=True)
    s.add_argument("--device", default="cuda")
    c = sub.add_parser("select")
    for k in ("outdir", "species", "xyz", "frames_dir"):
        c.add_argument(k)
    c.add_argument("--lo", type=float, default=0.1); c.add_argument("--hi", type=float, default=2.0)
    c.add_argument("--per-traj", type=int, default=10); c.add_argument("--gap-ps", type=float, default=0.5)
    c.add_argument("--budget", type=int, default=4000); c.add_argument("--stable-frac", type=float, default=0.3)
    a = ap.parse_args()
    {"jobs": cmd_jobs, "explore": cmd_explore, "score": cmd_score, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
