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


def pahdb_scan(path, uids, labels=False):
    """One pass over the PAHdb XML: {uid: array (unscaled freq, intensity)} for the wanted uids, sorted by unscaled
    frequency (the range-dependent scale factors can reorder the file's scaled order); with labels=True,
    {uid: (array, irrep labels in the same order)}."""
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
                modes = re.findall(r'<frequency scale="([\d.]+)">([-\d.]+)</frequency>\s*<intensity>([-\d.]+)</intensity>'
                                   r'\s*<symmetry>([^<]+)</symmetry>', s)
                arr = np.array([[float(f) / float(sc), float(i)] for sc, f, i, _ in modes])
                o = np.argsort(arr[:, 0], kind="stable")
                lab = [modes[k][3].replace("&quot;", '"').upper() for k in o]
                out[int(on)] = (arr[o], lab) if labels else arr[o]
                on = None
                if len(out) == len(want):
                    break
    return out


# Symmetry-resolved comparison (symfreq). PAHdb labels each mode with its irrep (Gaussian, standard orientation), so
# frequencies can be paired within each irrep rather than over all modes: a model mode that crosses a neighbour of
# another irrep is then caught. Abelian groups only, with axis-aligned operations (D2h, D2, C2v, C2h, Cs, C2, Ci);
# the labels depend on which axis is "z", so every axis permutation is tried and the one whose vibrational irrep
# counts (from the geometry alone) equal PAHdb's label counts is used; species with no match, or with two matching
# permutations that label modes differently, fall back to sorted pairing and are flagged.
OPS = {"E": np.eye(3), "C2z": np.diag([-1., -1, 1]), "C2y": np.diag([-1., 1, -1]), "C2x": np.diag([1., -1, -1]),
       "i": -np.eye(3), "sxy": np.diag([1., 1, -1]), "sxz": np.diag([1., -1, 1]), "syz": np.diag([-1., 1, 1])}
TABLES = {   # (operations, {irrep: characters}); Mulliken conventions
    "D2h": (["E", "C2z", "C2y", "C2x", "i", "sxy", "sxz", "syz"],
            {"AG": [1, 1, 1, 1, 1, 1, 1, 1], "B1G": [1, 1, -1, -1, 1, 1, -1, -1], "B2G": [1, -1, 1, -1, 1, -1, 1, -1],
             "B3G": [1, -1, -1, 1, 1, -1, -1, 1], "AU": [1, 1, 1, 1, -1, -1, -1, -1],
             "B1U": [1, 1, -1, -1, -1, -1, 1, 1], "B2U": [1, -1, 1, -1, -1, 1, -1, 1],
             "B3U": [1, -1, -1, 1, -1, 1, 1, -1]}),
    "D2": (["E", "C2z", "C2y", "C2x"], {"A": [1, 1, 1, 1], "B1": [1, 1, -1, -1], "B2": [1, -1, 1, -1], "B3": [1, -1, -1, 1]}),
    "C2v": (["E", "C2z", "sxz", "syz"], {"A1": [1, 1, 1, 1], "A2": [1, 1, -1, -1], "B1": [1, -1, 1, -1], "B2": [1, -1, -1, 1]}),
    "C2h": (["E", "C2z", "i", "sxy"], {"AG": [1, 1, 1, 1], "BG": [1, -1, 1, -1], "AU": [1, 1, -1, -1], "BU": [1, -1, -1, 1]}),
    "Cs": (["E", "sxy"], {"A'": [1, 1], 'A"': [1, -1]}),
    "C2": (["E", "C2z"], {"A": [1, 1], "B": [1, -1]}),
    "Ci": (["E", "i"], {"AG": [1, 1], "AU": [1, -1]}),
    "C1": (["E"], {"A": [1]}),          # no symmetry: the same as sorted pairing
}


def _perm_of(op, pos, sym, tol=0.02):
    q = pos @ op.T
    d = np.linalg.norm(q[:, None] - pos[None], axis=-1)
    j = d.argmin(1)
    ok = (d[np.arange(len(pos)), j] < tol).all() and len(set(j)) == len(pos) and all(sym[a] == sym[b] for a, b in enumerate(j))
    return j if ok else None


def _rep(op, perm):
    """3N x 3N matrix of the operation acting on displacements: (R u)_perm(i) = op u_i."""
    n = len(perm); M = np.zeros((3 * n, 3 * n))
    for i, j in enumerate(perm):
        M[3 * j:3 * j + 3, 3 * i:3 * i + 3] = op
    return M


def sym_resolve(pos, sym, masses, Hmw, labels):
    """Pair model and PAHdb frequencies within irreps. Returns (per-mode model freq in PAHdb order, info)."""
    want = {}
    for l in labels:
        want[l] = want.get(l, 0) + 1
    group = next((g for g, (ops, t) in TABLES.items() if set(t) == set(want)), None)
    if group is None:
        return None, dict(sym="unsupported", pahdb_irreps=sorted(want))
    ops, table = TABLES[group]
    com = (masses[:, None] * pos).sum(0) / masses.sum(); p0 = pos - com
    sols = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        P = np.eye(3)[list(perm)]              # table frame = P @ file frame
        reps = []
        for o in ops:
            op = P.T @ OPS[o] @ P              # the table's operation expressed in the file frame
            pm = _perm_of(op, p0, sym)
            if pm is None:
                break
            reps.append(_rep(op, pm))
        if len(reps) < len(ops):
            continue
        sqm = np.repeat(np.sqrt(masses), 3)
        # translations and rotations (mass-weighted), to remove from each irrep block
        tr = [np.tile(np.eye(3)[k], len(p0)) * sqm for k in range(3)]
        tr += [np.cross(np.eye(3)[k], p0).ravel() * sqm for k in range(3)]
        Q, _ = np.linalg.qr(np.array(tr).T)
        Pvib = np.eye(len(Q)) - Q @ Q.T
        out = {}
        for lab, chi in table.items():
            Pr = sum(c * R for c, R in zip(chi, reps)) / len(ops)
            Pr = Pvib @ Pr @ Pvib                       # projector onto vibrations of this irrep
            w, U = np.linalg.eigh(Pr)
            B = U[:, w > 0.5]                           # orthonormal basis of the subspace
            if B.shape[1] != want.get(lab, 0):
                break
            e = np.linalg.eigvalsh(B.T @ Hmw @ B) if B.shape[1] else np.array([])
            out[lab] = np.sort(np.sign(e) * np.sqrt(np.abs(e)) * CM1)
        else:
            sols.append((perm, out))
    if not sols:
        return None, dict(sym=group, status="no axis permutation reproduces the PAHdb irrep counts")
    status = "ok"
    if len(sols) > 1 and any(any(not np.allclose(sols[0][1][k], s[1][k]) for k in table) for s in sols[1:]):
        # Several axis choices reproduce the counts but label modes differently (e.g. D2 B1/B2/B3). PAHdb geometries
        # are in Gaussian's standard orientation (every unambiguous case so far selects the file's own axes), so use
        # those if they are among the solutions, and say so.
        if tuple(sols[0][0]) != (0, 1, 2):
            return None, dict(sym=group, status="ambiguous axis convention")
        status = "ok_file_axes"
    out = sols[0][1]; f = np.empty(len(labels)); idx = {}
    for k, l in enumerate(labels):
        idx.setdefault(l, []).append(k)
    for l, ks in idx.items():
        ks = sorted(ks)                     # PAHdb lists modes in ascending frequency
        f[ks] = out[l]
    return f, dict(sym=group, status=status, perm=list(sols[0][0]))


def symfreq(a):
    """Like freq, with symmetry-resolved pairing; writes per-mode tables (model, PAHdb, irrep)."""
    rows = [r for r in csv.DictReader(open(a.species)) if r["split"] in a.splits.split(",")]
    if a.pairs:
        rows = [r for r in rows if r["pair"] in a.pairs.split(",")]
    jobs = [(r["pair"], q, int(r[f"uid_q{q}"]), r["split"]) for r in rows for q in (0, 1)]
    ref = pahdb_scan(a.pahdb, [u for *_, u, _ in jobs], labels=True)
    calcs = {q: calculator(a.model, q, a.device) for q in (0, 1)}
    res = []
    for pair, q, uid, split in jobs:
        at = atoms_from(f"{a.xyz}/{pair}_q{q}.xyz"); at.calc = calcs[q]
        rel = relax(at, a.fmax)
        n = len(at); sym = at.get_chemical_symbols()
        m = np.array([MASS[s] for s in sym]); mm = np.repeat(m, 3)
        H = np.asarray(calcs[q].get_hessian(at)).reshape(3 * n, 3 * n); H = 0.5 * (H + H.T)
        Hmw = H / np.sqrt(np.outer(mm, mm))
        R, labels = ref[uid]
        f_sorted = harmonic(at, calcs[q])
        f_sym, info = sym_resolve(at.positions, sym, m, Hmw, labels)
        strong = R[:, 1] > 0.1 * R[:, 1].max()
        d_sorted = f_sorted - R[:, 0]
        r = dict(pair=pair, charge=q, uid=uid, split=split, n_atoms=n, **rel, **info,
                 rms_strong_sorted=float(np.sqrt((d_sorted[strong] ** 2).mean())),
                 n_imag=int((f_sorted < 0).sum()), imag=[float(x) for x in f_sorted[f_sorted < 0]],
                 freq_sorted=f_sorted.tolist(), freq_pahdb=R[:, 0].tolist(), inten_pahdb=R[:, 1].tolist(),
                 labels=labels, strong=strong.tolist())
        if f_sym is not None:
            d = f_sym - R[:, 0]
            r.update(rms_strong_sym=float(np.sqrt((d[strong] ** 2).mean())), maxabs_strong_sym=float(np.abs(d[strong]).max()),
                     freq_sym=f_sym.tolist(), imag_irreps=sorted({labels[k] for k in np.nonzero(f_sym < 0)[0]}))
        res.append(r)
        print(f"{pair:14s} q{q} {split:13s} {info.get('sym', ''):4s} {info.get('status', ''):10s} strong RMS sorted="
              f"{r['rms_strong_sorted']:.1f} sym={r.get('rms_strong_sym', float('nan')):.1f} imag={r['n_imag']} "
              f"{r.get('imag_irreps', '')}", flush=True)
        json.dump(res, open(a.out, "w"))


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
    try:                # several workers (e.g. one per GPU) may share a job list: first to take the lock runs it
        os.close(os.open(f"{a.outdir}/{tag}.lock", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        print(f"{tag}: locked by another worker (delete {tag}.lock if that worker died)", flush=True); return
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
    if a.traj_every:
        from ase.io.trajectory import Trajectory
        traj = Trajectory(f"{a.outdir}/{tag}.traj", "w", at, properties=[])
        nvt.attach(traj.write, interval=a.traj_every)
    nvt.run(n_nvt)
    nve = VelocityVerlet(at, dt)
    nve.attach(lambda: sample("nve", nve.nsteps), interval=100)
    if a.traj_every:
        nve.attach(traj.write, interval=a.traj_every)
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
    for name in ("freq", "symfreq"):
        f = sub.add_parser(name)
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
    m.add_argument("--traj-every", type=int, default=0, help="also write every N-th step to <tag>.traj (ASE)")
    a = ap.parse_args()
    {"freq": freq, "symfreq": symfreq, "md": md}[a.cmd](a)


if __name__ == "__main__":
    main()
