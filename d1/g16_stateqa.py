# Electronic-state QA side set for the charged-state MLFF data (committee review M2; docs/plan/charged_mlff_plan.md).
# The production jobs (g16_io.py write) use Gaussian's default initial guess. This checks, on a size-stratified sample,
# that the default SCF lands on the lowest solution, and how much a restricted open-shell (ROKS) treatment of the
# cations would change forces and dipoles.
#   python g16_stateqa.py make <frames dir> <species.csv> <qa gjf dir> [--pairs 24] [--basis 6-31G*] [--nproc 8]
#       per chosen pair (stratified by 10-C size bin; naphthalene, coronene, circumcoronene always in) and per charge:
#       the minimum (min) and the hottest intact MD frame (md, graph_changed=0). Variants, each a two-step job
#       (Guess=<variant> Stable=Opt, then Force from that wavefunction, all unrestricted):
#         closed-shell neutral: mix   (UB3LYP Guess=Mix: is there a broken-symmetry solution below RB3LYP?)
#         doublet cation:       huckel, mix  (other starting guesses: same state as the default?)
#         cation, 3 named pairs: roks (ROB3LYP Force, one step: force and dipole change against UKS)
#       Job names: <frame job name>_qa_<variant>; the frame job name is the one g16_io.py write gives the same frame.
#   python g16_stateqa.py report <qa log dir> <parsed production extxyz> [--tol-ha 1e-5]
#       per QA job: E(variant) - E(default) in Ha, and for roks the RMS force and dipole differences. A variant lower
#       than the default by more than --tol-ha marks the frame's species and charge "state_not_lowest".
import argparse, csv, glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g16_io import HARTREE_EV, parse_log, read_extxyz  # noqa: E402

SCF = "SCF=(XQC,Conver=8,MaxConventionalCycles=64,MaxCycle=256)"
ROKS_PAIRS = {"330", "18", "37"}                          # neutral uids: naphthalene, coronene, circumcoronene


def write_job(path, sym, p, q, m, basis, variant, nproc, mem):
    name = os.path.basename(path)[:-4]
    head = f"%nprocshared={nproc}\n%mem={mem}\n"
    with open(path, "w") as g:
        if variant == "roks":
            g.write(head + f"#P ROB3LYP/{basis} Force NoSymm Int=UltraFine {SCF}\n\n{name}\n\n{q} {m}\n")
            g.writelines(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n" for s, (x, y, z) in zip(sym, p))
            g.write("\n"); return
        guess = {"mix": "Guess=Mix", "huckel": "Guess=Huckel"}[variant]
        g.write(head + f"%chk={name}.chk\n#P UB3LYP/{basis} {guess} Stable=Opt NoSymm Int=UltraFine {SCF}\n\n{name}\n\n{q} {m}\n")
        g.writelines(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n" for s, (x, y, z) in zip(sym, p))
        g.write(f"\n--Link1--\n{head}%chk={name}.chk\n#P UB3LYP/{basis} Force NoSymm Int=UltraFine {SCF} Geom=AllCheck Guess=Read\n\n")


def cmd_make(a):
    rows = list(csv.DictReader(open(a.species)))
    must = [r for r in rows if r["uid_q0"] in ROKS_PAIRS]
    rest = [r for r in rows if r["uid_q0"] not in ROKS_PAIRS]
    rng = np.random.default_rng(a.seed); chosen = list(must)
    bins = sorted({int(r["nC"]) // 10 for r in rest})
    per = max(1, (a.pairs - len(must)) // len(bins))
    for b in bins:
        pool = [r for r in rest if int(r["nC"]) // 10 == b]
        chosen += [pool[i] for i in rng.choice(len(pool), size=min(per, len(pool)), replace=False)]
    os.makedirs(a.outdir, exist_ok=True); jobs = []
    for r in chosen:
        for q in (0, 1):
            tag = f"{r['pair']}_q{q}"
            frames = read_extxyz(f"{a.frames}/{tag}.extxyz")
            pick = {"min": next(k for k, (i, _, _) in enumerate(frames) if i["config_type"] == "min")}
            md = [k for k, (i, _, _) in enumerate(frames) if i["config_type"].startswith("md_") and i["graph_changed"] == "0"]
            if md:
                pick["md"] = max(md, key=lambda k: float(frames[k][0]["sample_T"]))
            m = 1 if q == 0 else 2
            variants = ["mix"] if m == 1 else ["huckel", "mix"] + (["roks"] if r["uid_q0"] in ROKS_PAIRS else [])
            for _, k in pick.items():
                info, sym, p = frames[k]
                for v in variants:
                    path = f"{a.outdir}/{tag}_f{k:03d}_qa_{v}.gjf"
                    write_job(path, sym, p, q, m, a.basis, v, a.nproc, a.mem); jobs.append(path)
    open(f"{a.outdir}/list.txt", "w").write("\n".join(os.path.abspath(j) for j in jobs) + "\n")
    print(f"{len(chosen)} pairs, {len(jobs)} QA jobs in {a.outdir} (list.txt)")


def cmd_report(a):
    ref = {}
    with open(a.parsed) as f:                            # production results: energy, dipole and forces per frame
        while (line := f.readline()).strip():
            n = int(line); info = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', f.readline()))
            F = np.array([[float(x) for x in f.readline().split()[4:7]] for _ in range(n)])
            ref[info["log"].rsplit(".", 1)[0]] = dict(E=float(info["dft_energy"]), F=F,
                                                      mu=np.array([float(x) for x in info["dft_dipole"].strip('"').split()]))
    rows = []
    for path in sorted(glob.glob(f"{a.qadir}/*_qa_*.log")):
        base = os.path.basename(path)[:-4]; frame, v = base.split("_qa_")
        try:
            r, err = parse_log(path)
        except Exception as e:
            r, err = None, f"parse error: {e}"
        if r is None:
            rows.append(dict(frame=frame, variant=v, status=err)); continue
        if frame not in ref:
            rows.append(dict(frame=frame, variant=v, status="no production result for this frame")); continue
        dE = (r["energy"] - ref[frame]["E"]) / HARTREE_EV
        row = dict(frame=frame, variant=v, status="ok", dE_Ha=f"{dE:+.2e}", s2=f"{r['s2']:.4f}", stability=r["stability"],
                   dmu_eA=f"{np.linalg.norm(r['dipole'] - ref[frame]['mu']):.4f}")
        if v == "roks":
            row["dF_rms_eVA"] = f"{np.sqrt(((r['F'] - ref[frame]['F']) ** 2).mean()):.4f}"
        elif dE < -a.tol_ha:
            row["flag"] = "state_not_lowest"
        rows.append(row)
    keys = ["frame", "variant", "status", "dE_Ha", "s2", "stability", "dmu_eA", "dF_rms_eVA", "flag"]
    w = csv.DictWriter(sys.stdout, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    bad = sorted({r["frame"].rsplit("_f", 1)[0] for r in rows if r.get("flag")})
    print(f"# {len(rows)} QA jobs; species/charge with a lower-energy solution than the default: {bad or 'none'}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("frames"); m.add_argument("species"); m.add_argument("outdir")
    m.add_argument("--pairs", type=int, default=24); m.add_argument("--basis", default="6-31G*")
    m.add_argument("--nproc", type=int, default=8); m.add_argument("--mem", default="16GB"); m.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("report"); r.add_argument("qadir"); r.add_argument("parsed"); r.add_argument("--tol-ha", type=float, default=1e-5)
    a = ap.parse_args()
    (cmd_make if a.cmd == "make" else cmd_report)(a)


if __name__ == "__main__":
    main()
