# Batched MLMD with the D1 force model (multi-head ScaleShiftMACE, head q<charge>) and a PolarMACE dipole model evaluated
# forward-only (compute_force=False) inside the loop: finite-T IR band strengths, shifts and widths for the
# coordinator's 2026-11-24 17:03 diagnostics (feat-dipole steps 2 and 3), in the format of the cloud-aimd DFT MD
# (d1/aimd_pyscf.py on cloud-aimd), so d1/aimd_slopes.py and d1/mode_energy.py analyse read the outputs unchanged.
# Protocol as aimd_pyscf.py: Maxwell-Boltzmann start at T (per-replica seed), Langevin NVT (friction 0.02 per ASE time
# unit; BAOAB here, ASE's Langevin there), centre of mass fixed (velocity removed every step, 3N-3 degrees of freedom in
# T), then velocity-Verlet NVE, 0.5 fs, ASE masses. Dipole about the centre of mass (isotopic masses, as the labels) every
# --dipole-fs (default 2 fs), e*A; NVE positions and velocities every --frames-fs (default 10 fs) for mode energies;
# (step, T, E_tot, 0, 0) every 20 steps in qa (columns of aimd_pyscf.py; <S^2> and SCF cycles do not apply).
# All replicas and temperatures of one species run as one batch of disjoint graphs (radius graph rebuilt every step
# on the GPU; no periodicity).
#   python mlmd_polar.py run <xyz or g5 npz> <out dir> --charge 1 --force v7.model --dipole polar.model
#       [--temps 300,900] [--replicas 32] [--nvt-ps 2] [--nve-ps 25] [--tag naph] [--seed0 0] [--dtype float32]
#   -> <out dir>/<tag>_q<c>_<T>K_r<k>.npz (keys as aimd_pyscf.py: dipole, qa, frames_pos, frames_vel, frames_every,
#      positions, velocities, step, nvt_ps, nve_ps, dt_fs (= --dipole-fs), T, seed, charge, mult, engines, seconds)
#   python mlmd_polar.py ref <xyz or g5 npz> <ref.npz> --charge 1 --force v7.model --dipole polar.model
#   -> harmonic reference in d1/mode_energy.py's format at the force model's minimum: modes from its Hessian, IR
#      intensities from central differences of the dipole model along them (ASE masses, as the MD)
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KM_FACTOR = 42.2561                       # km/mol per (D / (A amu^1/2))^2 (mode_energy.py)
D_PER_EA = 4.80320471
ISO = {1: 1.00782503, 6: 12.0}            # isotopic masses of the dipole labels (g16_io.MASS)


def load_geo(path):
    from ase import Atoms
    from ase.io import read
    if path.endswith(".npz"):                # dipole_g5.py run output: the force model's minimum
        d = np.load(path); m = d["masses"]
        num = np.where(m < 2, 1, 6)
        return Atoms(numbers=num, positions=d["pos"])
    return read(path)


class Graphs:
    """Template batch of B copies of one molecule for a mace model; positions and the radius graph are replaced per call."""

    def __init__(self, calc, atoms, B, charge, mult, device):
        import torch
        from mace import data as mace_data
        from mace.tools import torch_geometric
        self.torch = torch; self.calc = calc; self.model = calc.models[0]; self.B = B; self.N = len(atoms)
        keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
        at = atoms.copy(); at.info.update(charge=charge, spin=mult)
        cfg = mace_data.config_from_atoms(at, key_specification=keyspec, head_name=calc.head)
        g = mace_data.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
        batch = next(iter(torch_geometric.dataloader.DataLoader([g] * B, batch_size=B, shuffle=False))).to(device)
        self.d = batch.to_dict(); self.r_max = float(calc.r_max)
        self.dtype = self.d["positions"].dtype
        self.eye = torch.eye(self.N, dtype=torch.bool, device=device)

    def set(self, pos):
        """pos: (B, N, 3) tensor"""
        torch = self.torch
        with torch.no_grad():
            dist = torch.cdist(pos, pos)
            b, i, j = torch.nonzero((dist < self.r_max) & ~self.eye, as_tuple=True)
            off = b * self.N
            ei = torch.stack([off + i, off + j])
        self.d["positions"] = pos.reshape(-1, 3).to(self.dtype)
        self.d["edge_index"] = ei
        self.d["shifts"] = torch.zeros((ei.shape[1], 3), dtype=self.dtype, device=pos.device)
        self.d["unit_shifts"] = torch.zeros_like(self.d["shifts"])
        return self.d


def calcs(a, device):
    from mace.calculators import MACECalculator
    import polar_fastgraph; polar_fastgraph.install()
    fc = None
    if a.force:
        fc = MACECalculator(model_paths=a.force, device=device, default_dtype=a.dtype, head=f"q{a.charge}")
        assert fc.head == f"q{a.charge}", fc.head
    dc = MACECalculator(model_paths=a.dipole, device=device, default_dtype=a.dtype, model_type="PolarMACE")
    return fc, dc


def run(a):
    import torch
    from ase import units
    device = "cuda"
    atoms = load_geo(a.geo); N = len(atoms); mult = 1 if a.charge == 0 else 2
    temps = [float(t) for t in a.temps.split(",")]
    systems = [(T, r) for T in temps for r in range(a.replicas)]; B = len(systems)
    fc, dc = calcs(a, device)
    FG = Graphs(fc, atoms, B, a.charge, mult, device); DG = Graphs(dc, atoms, B, a.charge, mult, device)
    tdt = torch.float64
    m = torch.tensor(atoms.get_masses(), dtype=tdt, device=device)                 # ASE masses (dynamics)
    miso = torch.tensor([ISO[z] for z in atoms.numbers], dtype=tdt, device=device)  # isotopic (dipole COM)
    kB = units.kB; dt = 0.5 * units.fs; gamma = 0.02
    Tt = torch.tensor([T for T, _ in systems], dtype=tdt, device=device)
    seeds = [int(T) * 10 + 2 * r + a.charge + a.seed0 * 100000 for T, r in systems]
    x = torch.tensor(atoms.positions, dtype=tdt, device=device).expand(B, N, 3).clone()
    v = torch.empty((B, N, 3), dtype=tdt, device=device)
    for k, s in enumerate(seeds):           # Maxwell-Boltzmann at T, per-replica seed
        rng = np.random.default_rng(s)
        v[k] = torch.tensor(rng.standard_normal((N, 3)), dtype=tdt, device=device) * torch.sqrt(kB * Tt[k] / m)[:, None]
    gen = torch.Generator(device=device); gen.manual_seed(a.seed0 * 7919 + a.charge)

    def fix_com(v):
        return v - (m[None, :, None] * v).sum(1, keepdim=True) / m.sum()

    def forces(x):
        out = fc.models[0](FG.set(x), training=False, compute_force=True)
        return out["forces"].detach().to(tdt).reshape(B, N, 3), out["energy"].detach().to(tdt)

    def dipole(x):
        c = (miso[None, :, None] * x).sum(1, keepdim=True) / miso.sum()
        with torch.no_grad():
            out = dc.models[0](DG.set(x - c), training=False, compute_force=False)
        return out["dipole"].detach().to(tdt).reshape(B, 3)

    def temp(v):
        return (m[None, :, None] * v ** 2).sum((1, 2)) / ((3 * N - 3) * kB)

    n_nvt, n_nve = int(round(a.nvt_ps * 2000)), int(round(a.nve_ps * 2000))
    dstride = int(round(a.dipole_fs * 2)); fstride = int(round(a.frames_fs * 2))
    mu = torch.empty((n_nve // dstride, B, 3), dtype=torch.float64, device=device)
    fpos = np.empty((n_nve // fstride, B, N, 3), np.float32); fvel = np.empty_like(fpos)
    qa = []
    v = fix_com(v); f, e = forces(x)
    c1 = np.exp(-gamma * dt); c2 = torch.sqrt((1 - c1 ** 2) * kB * Tt / 1.0)[:, None, None] / torch.sqrt(m)[None, :, None]
    t0 = time.time()
    for step in range(1, n_nvt + n_nve + 1):
        nvt = step <= n_nvt
        v = v + 0.5 * dt * f / m[None, :, None]
        if nvt:                              # BAOAB: B A O A B
            x = x + 0.5 * dt * v
            v = c1 * v + c2 * torch.randn(v.shape, generator=gen, dtype=tdt, device=device)
            v = fix_com(v)
            x = x + 0.5 * dt * v
        else:
            x = x + dt * v
        f, e = forces(x)
        v = v + 0.5 * dt * f / m[None, :, None]
        if nvt:
            v = fix_com(v)
        s = step - n_nvt
        if s > 0 and s % dstride == 0:
            mu[s // dstride - 1] = dipole(x)
        if s > 0 and s % fstride == 0:
            fpos[s // fstride - 1] = x.cpu().numpy(); fvel[s // fstride - 1] = v.cpu().numpy()
        if step % 20 == 0:
            ek = 0.5 * (m[None, :, None] * v ** 2).sum((1, 2))
            qa.append(torch.stack([torch.full_like(ek, step), temp(v), e + ek, torch.zeros_like(ek), torch.zeros_like(ek)], 1).cpu().numpy())
        if step % 10000 == 0:
            print(f"step {step}/{n_nvt + n_nve} {time.time() - t0:.0f} s  T mean {temp(v).mean().item():.0f} K", flush=True)
    wall = time.time() - t0
    qa = np.stack(qa, 1)                     # (B, n_qa, 5)
    mu = mu.cpu().numpy(); xf = x.cpu().numpy(); vf = v.cpu().numpy()
    os.makedirs(a.out, exist_ok=True)
    for k, (T, r) in enumerate(systems):
        np.savez(f"{a.out}/{a.tag}_q{a.charge}_{int(T)}K_r{r}.npz", dipole=mu[:, k], qa=qa[k], frames_pos=fpos[:, k],
                 frames_vel=fvel[:, k], frames_every=fstride, positions=xf[k], velocities=vf[k], step=n_nvt + n_nve,
                 nvt_ps=a.nvt_ps, nve_ps=a.nve_ps, dt_fs=a.dipole_fs, T=T, seed=seeds[k], charge=a.charge, mult=mult,
                 engines=f"mlmd_polar force={os.path.basename(a.force)} dipole={a.dipole} {a.dtype}", seconds=wall / B,
                 method="MLMD v7 + PolarMACE", xyz=a.geo,
                 # aimd_slopes.load asserts step == n_nvt + 2 * len(dipole) (1 fs dipoles); store the equivalent
                 )
    json.dump(dict(B=B, N=N, steps=n_nvt + n_nve, wall_s=wall, ms_per_step=wall / (n_nvt + n_nve) * 1e3,
                   replica_steps_per_s=B * (n_nvt + n_nve) / wall), open(f"{a.out}/{a.tag}_q{a.charge}_timing.json", "w"))
    print(f"done: B={B} N={N} {n_nvt + n_nve} steps in {wall:.0f} s ({wall / (n_nvt + n_nve) * 1e3:.1f} ms/step, "
          f"{B * (n_nvt + n_nve) / wall:.0f} replica-steps/s)", flush=True)


def ref(a):
    from ase.optimize import BFGS
    device = "cuda"; a.dtype = "float64"
    atoms = load_geo(a.geo); N = len(atoms); mult = 1 if a.charge == 0 else 2
    fc, dc = calcs(a, device)
    atoms.calc = fc; BFGS(atoms, logfile=None).run(fmax=1e-4, steps=2000)
    m = atoms.get_masses(); pos = atoms.positions - (m[:, None] * atoms.positions).sum(0) / m.sum(); atoms.positions = pos
    H = np.asarray(fc.get_hessian(atoms)).reshape(3 * N, 3 * N); H = 0.5 * (H + H.T)
    sm = np.repeat(np.sqrt(m), 3); Hm = H / np.outer(sm, sm)
    T = []
    for i in range(3):
        u = np.zeros((N, 3)); u[:, i] = 1; T.append((u * np.sqrt(m)[:, None]).ravel())
    for i in range(3):
        e = np.zeros(3); e[i] = 1; T.append((np.cross(e, pos) * np.sqrt(m)[:, None]).ravel())
    Q, _ = np.linalg.qr(np.array(T).T); P = np.eye(3 * N) - Q @ Q.T
    w2, L = np.linalg.eigh(P @ Hm @ P)
    keep = np.argsort(np.abs(w2))[6:]; keep = keep[np.argsort(w2[keep])]; w2, L = w2[keep], L[:, keep]
    w_si = np.sqrt(np.abs(w2) * 1.602176634e-19 / 1.66053906660e-27) * 1e10
    wn = np.sign(w2) * w_si / (2 * np.pi * 2.99792458e10)
    miso = np.array([ISO[z] for z in atoms.numbers])

    def dip(p):
        b = atoms.copy(); b.calc = None; b.positions = p - (miso[:, None] * p).sum(0) / miso.sum()
        b.info.update(charge=a.charge, spin=mult)
        return np.asarray(dc.get_property("dipole", b)) * D_PER_EA          # Debye (e*A in)
    h = 0.01; dmu = []
    for k in range(L.shape[1]):
        d = (L[:, k] / sm).reshape(N, 3) * h
        dmu.append((dip(pos + d) - dip(pos - d)) / (2 * h))
    dmu = np.array(dmu); I = KM_FACTOR * (dmu ** 2).sum(1)
    np.savez(a.out, symbols=np.array(atoms.get_chemical_symbols()), masses=m, pos=pos, e0=atoms.get_potential_energy(), s2=0.0,
             w2=w2, wn=wn, L=L, dmu=dmu, I=I, charge=a.charge, mult=mult, gmax=float(np.abs(atoms.get_forces()).max()),
             method=f"force {os.path.basename(a.force)} head q{a.charge} Hessian; dipole {a.dipole} central differences")
    print(f"{a.out}: {N} atoms, lowest wn {wn[:3].round(1).tolist()}, total IR {I.sum():.1f} km/mol, max |F| {np.abs(atoms.get_forces()).max():.1e}")


def relabel(a):
    """Dipole model on the saved NVE frames of DFT MD runs (aimd_pyscf.py --frames-fs): writes <out>/model/<run>.npz with
    the model dipole series and <out>/dft/<run>.npz with the DFT dipole series at the same frames (both every
    frames_every steps), so the dipole model is scored on identical dynamics (no force-model difference)."""
    import torch
    from ase import Atoms
    device = "cuda"
    _, dc = calcs(argparse.Namespace(force=a.force, dipole=a.dipole, dtype=a.dtype, charge=a.charge), device)
    for sub in ("model", "dft"):
        os.makedirs(f"{a.out}/{sub}", exist_ok=True)
    for p in a.runs:
        z = dict(np.load(p)); fr = int(z["frames_every"]); P = np.asarray(z["frames_pos"], float)
        num = load_geo(str(z["xyz"]) if os.path.exists(str(z["xyz"])) else a.xyz).numbers
        assert len(num) == P.shape[1]
        at = Atoms(numbers=num, positions=P[0]); mult = int(z["mult"])
        G = Graphs(dc, at, len(P), int(z["charge"]), mult, device)
        miso = torch.tensor([ISO[x] for x in num], dtype=torch.float64, device=device)
        x = torch.tensor(P, dtype=torch.float64, device=device); c = (miso[None, :, None] * x).sum(1, keepdim=True) / miso.sum()
        with torch.no_grad():
            mu = dc.models[0](G.set(x - c), training=False, compute_force=False)["dipole"].double().cpu().numpy()
        # DFT dipoles: every 2 steps (1 fs), the k-th at NVE step 2(k+1); frame j at NVE step fr(j+1)
        idx = np.arange(1, len(P) + 1) * fr // 2 - 1
        dft = np.asarray(z["dipole"])[idx]
        base = dict(z, dt_fs=fr * 0.5)
        for sub, d in (("model", mu), ("dft", dft)):
            np.savez(f"{a.out}/{sub}/{os.path.basename(p)}", **dict(base, dipole=d))
        print(f"{os.path.basename(p)}: {len(P)} frames every {fr * 0.5} fs; mean |mu_model - mu_DFT| "
              f"{np.linalg.norm(mu - dft, axis=1).mean() * D_PER_EA:.3f} D", flush=True)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    rl = sp.add_parser("relabel"); rl.add_argument("out"); rl.add_argument("runs", nargs="+")
    rl.add_argument("--charge", type=int, required=True); rl.add_argument("--dipole", required=True)
    rl.add_argument("--force", default=None); rl.add_argument("--dtype", default="float32")
    rl.add_argument("--xyz", help="geometry file giving the atom order, if the run's own xyz path does not exist here")
    for name in ("run", "ref"):
        p = sp.add_parser(name); p.add_argument("geo"); p.add_argument("out")
        p.add_argument("--charge", type=int, required=True); p.add_argument("--force", required=True)
        p.add_argument("--dipole", required=True); p.add_argument("--dtype", default="float32")
    r = sp.choices["run"]
    r.add_argument("--temps", default="300,900"); r.add_argument("--replicas", type=int, default=32)
    r.add_argument("--nvt-ps", type=float, default=2.0); r.add_argument("--nve-ps", type=float, default=25.0)
    r.add_argument("--dipole-fs", type=float, default=2.0); r.add_argument("--frames-fs", type=float, default=10.0)
    r.add_argument("--tag", default="naph"); r.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    {"run": run, "ref": ref, "relabel": relabel}[a.cmd](a)


if __name__ == "__main__":
    main()
