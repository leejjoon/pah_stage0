# Acceptance checks for a mace-torch charge-state model (docs/plan/charged_mlff_proposal.md §5.7, G4 and G5), run in
# container B. One head per charge state: q0 (neutral) and q1 (cation); the head is asserted, never defaulted.
#
# freq (G5, frequency half): for each pair and charge, start from the PAHdb geometry (mlff_species.py xyz), relax with
#   the model (float64, BFGS to --fmax), take the analytic model Hessian, and compare the harmonic frequencies with
#   PAHdb v4.00 (B3LYP/6-31G*, Gaussian; scale factors divided out), mode by mode in sorted order. Strong bands are
#   those with PAHdb intensity > 10 % of the strongest; the G5 criterion is their RMS deviation <= 10 cm^-1.
#   Intensities need a dipole model, which this model lacks: the intensity half of G5 is not tested here.
#   Sorted-order pairing can swap near-degenerate modes (dense 1100-1600 cm^-1 region), which inflates the deviation;
#   the per-mode table is kept so swaps can be checked.
#     python mlff_accept.py freq <model> <species_v2.csv> <xyz dir> <pahdb.xml> <out.json> [--splits validation,...]
#                                [--pairs A,B] [--exclude A,B] [--device cpu] [--fmax 1e-4]
#
# md (G4 subset): ASE Langevin NVT then NVE (velocity Verlet), 0.5 fs, friction 0.02 per ASE time unit as in
#   d1/mlmd_run.py, COM momentum removed; Maxwell-Boltzmann start at T on the model-relaxed PAHdb geometry. Logs
#   T and E_tot every 100 steps, and compares the final bond graph with the input (d1/mlmd_integrity.py bonds()).
#     python mlff_accept.py md <model> <xyz dir> <out dir> --pair P --charge 0 --temp 900 --seed 0
#                              [--nvt-ps 20] [--nve-ps 20] [--device cuda]
import argparse, csv, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_integrity import bonds, describe, read_xyz

CM1 = 521.47091         # sqrt(eV / (A^2 amu)) in cm^-1
MASS = {"H": 1.00782503, "C": 12.0}     # most abundant isotope, as Gaussian/PAHdb


def calculator(model, charge, device, dtype="float64"):
    from mace.calculators import MACECalculator
    c = MACECalculator(model_paths=model, device=device, default_dtype=dtype, head=f"q{charge}")
    assert c.available_heads == ["q0", "q1"] and c.head == f"q{charge}", (c.available_heads, c.head)
    return c


def atoms_from(xyz):
    from ase import Atoms
    sym, pos = read_xyz(xyz)
    return Atoms(sym, positions=pos)


def relax(atoms, fmax, steps=5000):
    from ase.optimize import BFGS
    p0 = atoms.positions.copy()
    opt = BFGS(atoms, logfile=None, maxstep=0.05)
    ok = opt.run(fmax=fmax, steps=steps)
    d = atoms.positions - p0; d -= d.mean(0)
    return dict(converged=bool(ok), steps=opt.nsteps, fmax=float(np.abs(atoms.get_forces()).max()),
                max_disp=float(np.linalg.norm(d, axis=1).max()), rmsd=float(np.sqrt((d ** 2).sum(1).mean())))


def harmonic(atoms, calc):
    n = len(atoms)
    H = np.asarray(calc.get_hessian(atoms)).reshape(3 * n, 3 * n)
    H = 0.5 * (H + H.T)
    m = np.repeat([MASS[s] for s in atoms.get_chemical_symbols()], 3)
    w = np.linalg.eigvalsh(H / np.sqrt(np.outer(m, m)))
    f = np.sign(w) * np.sqrt(np.abs(w)) * CM1
    f = f[np.argsort(np.abs(f))][6:]            # drop the 6 translations/rotations (nonlinear)
    return np.sort(f)


def pahdb_scan(path, uids):
    """One pass over the PAHdb XML: {uid: (unscaled freq, intensity)} for the wanted uids."""
    want, out, buf, on = {str(u) for u in uids}, {}, [], None
    for line in open(path):
        if on is None:
            m = re.search(r'<specie uid="(\d+)"', line)
            if m and m.group(1) in want:
                on, buf = m.group(1), []
        if on is not None:
            buf.append(line)
            if "</specie>" in line:
                s = "".join(buf)
                modes = re.findall(r'<frequency scale="([\d.]+)">([-\d.]+)</frequency>\s*<intensity>([-\d.]+)</intensity>', s)
                out[int(on)] = np.array([[float(f) / float(sc), float(i)] for sc, f, i in modes])
                on = None
                if len(out) == len(want):
                    break
    return out


def freq(a):
    rows = [r for r in csv.DictReader(open(a.species)) if r["split"] in a.splits.split(",")]
    if a.pairs:
        rows = [r for r in rows if r["pair"] in a.pairs.split(",")]
    rows = [r for r in rows if r["pair"] not in a.exclude.split(",")]
    jobs = [(r["pair"], q, int(r[f"uid_q{q}"]), r["split"]) for r in rows for q in (0, 1)]
    ref = pahdb_scan(a.pahdb, [u for *_, u, _ in jobs])
    calcs = {q: calculator(a.model, q, a.device) for q in (0, 1)}
    res = []
    for pair, q, uid, split in jobs:
        t = time.time()
        at = atoms_from(f"{a.xyz}/{pair}_q{q}.xyz"); at.calc = calcs[q]
        f0 = float(np.abs(at.get_forces()).max())
        rel = relax(at, a.fmax)
        f = harmonic(at, calcs[q]); R = ref[uid]
        if len(R) != len(f):
            print(f"{pair} q{q}: {len(f)} model modes vs {len(R)} PAHdb", flush=True); continue
        df = f - R[:, 0]; strong = R[:, 1] > 0.1 * R[:, 1].max()
        r = dict(pair=pair, charge=q, uid=uid, split=split, n_atoms=len(at), fmax_pahdb_geom=f0, **rel,
                 n_imag=int((f < 0).sum()), rms_strong=float(np.sqrt((df[strong] ** 2).mean())),
                 maxabs_strong=float(np.abs(df[strong]).max()), n_strong=int(strong.sum()),
                 rms_all=float(np.sqrt((df ** 2).mean())), mad_all=float(np.abs(df).mean()),
                 freq=f.tolist(), freq_pahdb=R[:, 0].tolist(), inten_pahdb=R[:, 1].tolist())
        res.append(r)
        print(f"{pair:14s} q{q} {split:13s} N={len(at):3d} F@PAHdb={f0:.3f} relax rmsd={rel['rmsd']:.3f} "
              f"imag={r['n_imag']} strong RMS={r['rms_strong']:.1f} max={r['maxabs_strong']:.1f} (n={r['n_strong']}) "
              f"all RMS={r['rms_all']:.1f}  {time.time() - t:.0f}s", flush=True)
        json.dump(res, open(a.out, "w"))


def md(a):
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    os.makedirs(a.outdir, exist_ok=True)
    tag = f"{a.pair}_q{a.charge}_T{a.temp}_s{a.seed}"
    if os.path.exists(f"{a.outdir}/{tag}.json"):
        return
    calc = calculator(a.model, a.charge, a.device, "float32")
    at = atoms_from(f"{a.xyz}/{a.pair}_q{a.charge}.xyz"); at.calc = calc
    relax(at, 1e-3, 2000)
    sym = at.get_chemical_symbols(); b0 = bonds(sym, at.positions)
    MaxwellBoltzmannDistribution(at, temperature_K=a.temp, rng=np.random.default_rng(a.seed)); Stationary(at)
    dt = 0.5 * units.fs
    log, t0 = [], time.time()

    def sample(phase, step):
        log.append((phase, step, float(at.get_temperature()), float(at.get_total_energy())))

    nvt = Langevin(at, dt, temperature_K=a.temp, friction=0.02, fixcm=True, rng=np.random.default_rng(a.seed + 1))
    n_nvt, n_nve = int(a.nvt_ps * 2000), int(a.nve_ps * 2000)
    nvt.attach(lambda: sample("nvt", nvt.nsteps), interval=100)
    nvt.run(n_nvt)
    nve = VelocityVerlet(at, dt)
    nve.attach(lambda: sample("nve", nve.nsteps), interval=100)
    nve.run(n_nve)
    b1 = bonds(sym, at.positions)
    E = np.array([e for p, _, _, e in log if p == "nve"]); T = np.array([t for p, _, t, _ in log if p == "nve"])
    r = dict(pair=a.pair, charge=a.charge, temp=a.temp, seed=a.seed, n_atoms=len(at), nvt_ps=a.nvt_ps, nve_ps=a.nve_ps,
             intact=b0 == b1, change=describe(sym, b1 - b0, b0 - b1) if b0 != b1 else "",
             nve_drift_meV_atom=float((E[-1] - E[0]) / len(at) * 1e3), nve_Estd_meV_atom=float(E.std() / len(at) * 1e3),
             nve_T_mean=float(T.mean()), wall_s=round(time.time() - t0, 1), log=log)
    from ase.io import write
    write(f"{a.outdir}/{tag}_final.xyz", at)
    json.dump(r, open(f"{a.outdir}/{tag}.json", "w"))
    print(f"{tag}: intact={r['intact']} {r['change']} drift={r['nve_drift_meV_atom']:.2f} meV/atom "
          f"T_nve={r['nve_T_mean']:.0f} K wall={r['wall_s']:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freq")
    for k in ("model", "species", "xyz", "pahdb", "out"):
        f.add_argument(k)
    f.add_argument("--splits", default="validation,final_test,transfer_test")
    f.add_argument("--pairs", default=""); f.add_argument("--exclude", default="")
    f.add_argument("--device", default="cpu"); f.add_argument("--fmax", type=float, default=1e-4)
    m = sub.add_parser("md")
    for k in ("model", "xyz", "outdir"):
        m.add_argument(k)
    m.add_argument("--pair", required=True); m.add_argument("--charge", type=int, required=True)
    m.add_argument("--temp", type=float, required=True); m.add_argument("--seed", type=int, default=0)
    m.add_argument("--nvt-ps", type=float, default=20); m.add_argument("--nve-ps", type=float, default=20)
    m.add_argument("--device", default="cuda")
    a = ap.parse_args()
    freq(a) if a.cmd == "freq" else md(a)


if __name__ == "__main__":
    main()
