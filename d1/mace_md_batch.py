# Batched MACE MLMD for D1 production (course correction 2026-11-23, plan "Status log"): one species, charge and
# temperature per call, B independent replicas on one GPU through TorchSim (container B_torchsim), instead of serial
# single-molecule ASE runs (mlff_accept.py md, ~95 steps/s/GPU).
#   python mace_md_batch.py run <model> <minimum.xyz> <out prefix> --charge Q --temp T [--replicas 32] [--nvt-ps 10]
#                               [--nve-ps 50] [--save-fs 25] [--dt-fs 0.5] [--gamma 10] [--seed 0] [--cueq] [--float64]
#   python mace_md_batch.py check <model> <minimum.xyz> --charge Q --temp T [--steps 400]
#   python mace_md_batch.py multi <model> <jobs.txt> <xyz dir> <out prefix> --charge Q [--cueq] [--nvt-ps 20] [--nve-ps 50]
#                               [--save-fs 10] [--gamma 10] [--seed 0] [--float64] [--parity K]
# multi: several species and temperatures in one batch (regression G4 / production sweeps). jobs.txt lines
#   "<pair> <T> <replicas>", minimum <xyz dir>/<pair>_q<Q>.xyz; per-system kT. NVT frames every 100 fs, NVE every
#   --save-fs; npz: pos [frames, all atoms, 3] with per-system offsets, epot, ekin, t_ps, temps, pairs. json per system:
#   intact (no bond-graph change vs the minimum held over the last 100 fs), change, first_change_ps, frames_changed,
#   blowup_ps (T > 3 x target), NVE drift, T_nve. --parity K: forces and energies of K saved NVE frame sets
#   recomputed with the other backend (cueq <-> plain e3nn), max |dF| and |dE|/atom.
# run: B copies of the minimum, Maxwell-Boltzmann velocities at T (per-replica seeds), Langevin NVT (BAOAB, friction
#   --gamma ps^-1) for --nvt-ps: the replicas share no noise, so their NVE starts are decorrelated snapshots; centre-of-
#   mass velocity of every replica set to zero at the start of NVE (no rotation removal; the COM policy of this runner);
#   NVE velocity Verlet for --nve-ps. Every --save-fs of NVE: positions and velocities (float32), potential and kinetic
#   energy per replica. Outputs <prefix>.npz (pos, vel [frames, B, N, 3], epot, ekin [frames, B] eV, symbols, masses,
#   dt_fs, save_fs) and <prefix>.json (settings, per-replica NVE drift in meV/atom over the run from a linear fit, mean
#   T, bond-graph integrity per replica as g4_score.py: changes against the start held over the last 100 fs, the first
#   change, blow-up time if T exceeds 3x the target; steps/s and atom-steps/s).
# intact: no persistent change over the last 100 fs, the NVE start graph equal to the minimum's (no change during NVT) and no
#   blow-up. Without --cueq, batches above 30,720 atoms are refused: at 1024 x 60 atoms the plain e3nn path gave
#   non-physical forces to replicas 318-1023 from the start (512 x 60 clean; all cueq runs to 1024 x 60 clean).
# check: the same initial state propagated by this runner's NVE and by ASE VelocityVerlet + MACECalculator (float64,
#   one replica), max |dx| after --steps; checks units (TorchSim takes dt in internal units here: dt_fs * ase.units.fs)
#   and the head.
# Heads: the model's head "q<charge>". Positions stay in vacuum (no cell, pbc off).
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_integrity import bonds, describe, read_xyz


def setup(a, replicas, dtype):
    import torch, torch_sim as ts
    from ase import Atoms
    from torch_sim.models.mace import MaceModel
    sym, pos = read_xyz(a.xyz)
    at = Atoms(sym, positions=pos, cell=np.eye(3) * 100.0, pbc=False)
    dev = torch.device("cuda")
    model = MaceModel(model=torch.load(a.model, map_location=dev, weights_only=False), device=dev, dtype=dtype,
                      compute_stress=False, enable_cueq=getattr(a, "cueq", False), head=f"q{a.charge}")
    state = ts.initialize_state([at.copy() for _ in range(replicas)], device=dev, dtype=dtype)
    return at, model, state


def zero_com(state):
    import torch
    n = int(state.system_idx.max()) + 1; m = state.masses[:, None]
    P = torch.zeros(n, 3, device=m.device, dtype=m.dtype).index_add_(0, state.system_idx, state.momenta)
    M = torch.zeros(n, device=m.device, dtype=m.dtype).index_add_(0, state.system_idx, state.masses)
    state.momenta = state.momenta - m * (P / M[:, None])[state.system_idx]
    return state


def per_system(x, state, n):
    import torch
    return torch.zeros(n, device=x.device, dtype=x.dtype).index_add_(0, state.system_idx, x)


def run(a):
    import torch, torch_sim as ts
    from ase import units
    dtype = torch.float64 if a.float64 else torch.float32
    n_at = len(read_xyz(a.xyz)[0]) * a.replicas
    if n_at > 30720 and not a.cueq:   # without cuEquivariance, 1024 x 60 atoms corrupted replicas >= 318 (T ~1e15 K from the first
        sys.exit(f"{n_at} atoms in the batch: use --cueq above 30720 atoms (the plain e3nn path corrupts large batches)")   # NVT steps); 512 x 60 was clean
    at, model, state = setup(a, a.replicas, dtype); B = a.replicas; N = len(at); sym = at.get_chemical_symbols()
    kT = a.temp * units.kB; dt = a.dt_fs * units.fs; gamma = a.gamma / (1000 * units.fs)      # internal units
    gen = torch.Generator(device="cuda"); gen.manual_seed(a.seed)
    state = ts.nvt_langevin_init(state, model, kT=kT)
    state.momenta = ts.integrators.initialize_momenta(state.positions, state.masses, state.system_idx,
                                                      torch.full((B,), kT, device="cuda", dtype=dtype), gen)
    n_nvt = int(round(a.nvt_ps * 1000 / a.dt_fs)); n_nve = int(round(a.nve_ps * 1000 / a.dt_fs))
    every = int(round(a.save_fs / a.dt_fs)); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n_nvt):
        state = ts.nvt_langevin_step(state, model, dt=dt, kT=kT, gamma=gamma)
    state = zero_com(state)          # the Langevin MDState goes on into NVE as is (nve_init would draw new momenta)
    m = state.masses
    POS, VEL, EP, EK = [], [], [], []
    def save(s):
        POS.append(s.positions.view(B, N, 3).float().cpu().numpy()); VEL.append((s.momenta / m[:, None]).view(B, N, 3).float().cpu().numpy())
        EP.append(s.energy.double().cpu().numpy()); EK.append(per_system(0.5 * (s.momenta ** 2).sum(1) / m, s, B).double().cpu().numpy())
    save(state)
    for k in range(1, n_nve + 1):
        state = ts.nve_step(state, model, dt=dt)
        if k % every == 0:
            save(state)
    torch.cuda.synchronize(); wall = time.time() - t0
    pos, vel = np.array(POS), np.array(VEL); ep, ek = np.array(EP), np.array(EK)
    np.savez_compressed(a.out + ".npz", pos=pos, vel=vel, epot=ep, ekin=ek, symbols=np.array(sym), masses=at.get_masses(),
                        dt_fs=a.dt_fs, save_fs=a.save_fs)
    t_ps = np.arange(len(ep)) * a.save_fs / 1000; dof = 3 * N - 3
    Tm = 2 * ek / (dof * units.kB); b0 = bonds(sym, pos[0, 0])
    win = max(1, int(round(100 / a.save_fs))); reps = []
    for r in range(B):
        etot = ep[:, r] + ek[:, r]; slope = np.polyfit(t_ps, etot, 1)[0]
        g = [frozenset(bonds(sym, pos[k, r])) for k in range(len(pos))]; b0r = frozenset(bonds(sym, pos[0, r]))
        lost = set.intersection(*(set(b0r - x) for x in g[-win:])); gained = set.intersection(*(set(x - b0r) for x in g[-win:]))
        first = next((t_ps[k] for k in range(len(g)) if g[k] != b0r), None)
        blow = next((t_ps[k] for k in range(len(g)) if Tm[k, r] > 3 * a.temp), None)
        reps.append(dict(replica=r, drift_meV_atom=1000 * slope * a.nve_ps / N, T_mean=float(Tm[:, r].mean()),
                         intact=not (lost or gained) and b0r == frozenset(b0) and blow is None, change=describe(sym, gained, lost) if (lost or gained) else "",
                         first_change_ps=first, blowup_ps=blow, start_graph_as_minimum=b0r == frozenset(b0)))
    steps = n_nvt + n_nve
    res = dict(model=a.model, xyz=a.xyz, charge=a.charge, temp=a.temp, replicas=B, atoms=N, nvt_ps=a.nvt_ps, nve_ps=a.nve_ps,
               dt_fs=a.dt_fs, save_fs=a.save_fs, gamma_ps=a.gamma, seed=a.seed, cueq=a.cueq, dtype=str(dtype), wall_s=wall,
               steps_per_s=steps / wall, replica_steps_per_s=B * steps / wall, atom_steps_per_s=B * N * steps / wall,
               intact=sum(r["intact"] for r in reps), replicas_detail=reps)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    d = [abs(r["drift_meV_atom"]) for r in reps]
    print(f"{os.path.basename(a.out)}: {B} x {N} atoms, {a.nvt_ps}+{a.nve_ps} ps, {wall:.0f} s, {steps / wall:.1f} steps/s "
          f"({B * steps / wall:.0f} replica-steps/s); intact {res['intact']}/{B}; |drift| median {np.median(d):.2f} max {max(d):.2f} meV/atom; "
          f"T_NVE mean {np.mean([r['T_mean'] for r in reps]):.0f} K")


def multi(a):
    # Several species and temperatures in one batch (one charge = one head): jobs file lines "<pair> <T> <replicas>",
    # minimum <xyz dir>/<pair>_q<charge>.xyz. Langevin NVT (per-system kT) then NVE as run(); frames every --save-fs
    # over NVE (and every 100 fs over NVT, for the first change). Per system, G4 scoring as g4_score.py, against the
    # species' minimum graph: intact = no persistent change over the last 100 fs (lost/gained bonds present in every
    # frame of it), plus first change, blow-up (T > 3x target) and NVE drift. --parity K: energies and forces of K
    # saved NVE frame sets (all systems) with the cuEquivariance and the plain e3nn backend, max |dF| and |dE|/atom.
    import torch, torch_sim as ts
    from ase import Atoms, units
    from torch_sim.models.mace import MaceModel
    dtype = torch.float64 if a.float64 else torch.float32; dev = torch.device("cuda")
    jobs = [l.split() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    ats, temps, meta = [], [], []
    for pair, T, n in jobs:
        sym, pos = read_xyz(f"{a.xyz_dir}/{pair}_q{a.charge}.xyz")
        for r in range(int(n)):
            ats.append(Atoms(sym, positions=pos, cell=np.eye(3) * 100.0, pbc=False)); temps.append(float(T))
            meta.append(dict(pair=pair, temp=float(T), replica=r, atoms=len(sym)))
    n_at = sum(len(x) for x in ats); B = len(ats)
    if n_at > 30720 and not a.cueq:
        sys.exit(f"{n_at} atoms in the batch: use --cueq above 30720 atoms")
    raw = torch.load(a.model, map_location=dev, weights_only=False)
    model = MaceModel(model=raw, device=dev, dtype=dtype, compute_stress=False, enable_cueq=a.cueq, head=f"q{a.charge}")
    state = ts.initialize_state(ats, device=dev, dtype=dtype)
    kT = torch.tensor(temps, device=dev, dtype=dtype) * units.kB
    dt = a.dt_fs * units.fs; gamma = a.gamma / (1000 * units.fs)
    gen = torch.Generator(device="cuda"); gen.manual_seed(a.seed)
    state = ts.nvt_langevin_init(state, model, kT=kT)
    state.momenta = ts.integrators.initialize_momenta(state.positions, state.masses, state.system_idx, kT, gen)
    n_nvt = int(round(a.nvt_ps * 1000 / a.dt_fs)); n_nve = int(round(a.nve_ps * 1000 / a.dt_fs))
    every = int(round(a.save_fs / a.dt_fs)); every_nvt = int(round(100 / a.dt_fs))
    m = state.masses; POS, EP, EK, TT = [], [], [], []
    def save(s, t_ps):
        POS.append(s.positions.float().cpu().numpy()); TT.append(t_ps)
        EP.append(s.energy.double().cpu().numpy()); EK.append(per_system(0.5 * (s.momenta ** 2).sum(1) / m, s, B).double().cpu().numpy())
    torch.cuda.synchronize(); t0 = time.time()
    for k in range(n_nvt):
        state = ts.nvt_langevin_step(state, model, dt=dt, kT=kT, gamma=gamma)
        if (k + 1) % every_nvt == 0 and k + 1 < n_nvt:
            save(state, (k + 1) * a.dt_fs / 1000)
    state = zero_com(state); n_save_nvt = len(POS); save(state, a.nvt_ps)
    for k in range(1, n_nve + 1):
        state = ts.nve_step(state, model, dt=dt)
        if k % every == 0:
            save(state, a.nvt_ps + k * a.dt_fs / 1000)
    torch.cuda.synchronize(); wall = time.time() - t0
    pos = np.array(POS); ep, ek, tps = np.array(EP), np.array(EK), np.array(TT)
    off = np.concatenate([[0], np.cumsum([x["atoms"] for x in meta])])
    np.savez_compressed(a.out + ".npz", pos=pos, epot=ep, ekin=ek, t_ps=tps, n_save_nvt=n_save_nvt, offsets=off,
                        temps=np.array(temps), pairs=np.array([x["pair"] for x in meta]), dt_fs=a.dt_fs, save_fs=a.save_fs)
    win = max(1, int(round(100 / a.save_fs))); nve = slice(n_save_nvt, None); res = []
    for i, x in enumerate(meta):
        sym = ats[i].get_chemical_symbols(); P = pos[:, off[i]:off[i + 1]]; N = x["atoms"]
        b0 = frozenset(bonds(sym, ats[i].positions)); g = [frozenset(bonds(sym, P[k])) for k in range(len(P))]
        lost = set.intersection(*(set(b0 - y) for y in g[-win:])); gained = set.intersection(*(set(y - b0) for y in g[-win:]))
        Tm = 2 * ek[:, i] / ((3 * N - 3) * units.kB)
        t_nve = tps[nve] - a.nvt_ps; etot = ep[nve, i] + ek[nve, i]
        x.update(intact=not (lost or gained), change=describe(sym, gained, lost) if (lost or gained) else "",
                 first_change_ps=next((float(tps[k]) for k in range(len(g)) if g[k] != b0), None),
                 frames_changed=float(np.mean([y != b0 for y in g])),
                 blowup_ps=next((float(tps[k]) for k in range(len(g)) if Tm[k] > 3 * x["temp"]), None),
                 drift_meV_atom=float(1000 * np.polyfit(t_nve, etot, 1)[0] * a.nve_ps / N), T_nve=float(Tm[nve].mean()))
        res.append(x)
    out = dict(model=a.model, jobs=a.jobs, charge=a.charge, systems=B, atoms=n_at, nvt_ps=a.nvt_ps, nve_ps=a.nve_ps,
               dt_fs=a.dt_fs, save_fs=a.save_fs, gamma_ps=a.gamma, seed=a.seed, cueq=a.cueq, dtype=str(dtype), wall_s=wall,
               steps_per_s=(n_nvt + n_nve) / wall, atom_steps_per_s=n_at * (n_nvt + n_nve) / wall, systems_detail=res)
    if a.parity:
        other = MaceModel(model=torch.load(a.model, map_location=dev, weights_only=False), device=dev, dtype=dtype,
                          compute_stress=False, enable_cueq=not a.cueq, head=f"q{a.charge}")
        dF, dE = [], []
        for k in np.linspace(n_save_nvt, len(pos) - 1, a.parity).astype(int):
            st = ts.initialize_state(ats, device=dev, dtype=dtype); st.positions = torch.tensor(pos[k], device=dev, dtype=dtype)
            o1, o2 = model(st), other(st)
            dF.append(float((o1["forces"] - o2["forces"]).abs().max())); dE.append(float(((o1["energy"] - o2["energy"]).abs() / torch.tensor([x["atoms"] for x in meta], device=dev)).max()))
        out["parity"] = dict(frames=a.parity, max_dF_eV_A=max(dF), max_dE_eV_atom=max(dE))
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    print(f"{os.path.basename(a.out)}: {B} systems, {n_at} atoms, {a.nvt_ps}+{a.nve_ps} ps, {wall:.0f} s, "
          f"{(n_nvt + n_nve) / wall:.1f} steps/s" + (f"; parity max |dF| {out['parity']['max_dF_eV_A']:.2e} eV/A, |dE| {out['parity']['max_dE_eV_atom']:.2e} eV/atom" if a.parity else ""))
    groups = {}
    for r in res:
        groups.setdefault((r["pair"], r["temp"]), []).append(r)
    for (pair, T), grp in sorted(groups.items()):
        bad = [r for r in grp if not r["intact"]]
        print(f"  {pair} q{a.charge} {T:.0f} K: intact {len(grp) - len(bad)}/{len(grp)}; |drift| max {max(abs(r['drift_meV_atom']) for r in grp):.2f} meV/atom"
              + "".join(f"; r{r['replica']}: {r['change']}" for r in bad))


def check(a):
    import torch, torch_sim as ts
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from mace.calculators import MACECalculator
    at, model, state = setup(a, 1, torch.float64)
    gen = torch.Generator(device="cuda"); gen.manual_seed(1)
    kT = a.temp * units.kB
    state = ts.nve_init(state, model, kT=kT)
    state.momenta = ts.integrators.initialize_momenta(state.positions, state.masses, state.system_idx,
                                                      torch.full((1,), kT, device="cuda", dtype=torch.float64), gen)
    state = zero_com(state)
    b = at.copy(); b.set_momenta(state.momenta.cpu().numpy())
    b.calc = MACECalculator(model_paths=a.model, device="cuda", default_dtype="float64", head=f"q{a.charge}")
    e_ts = float(model(state)["energy"][0]); e_ase = b.get_potential_energy()
    dyn = VelocityVerlet(b, timestep=a.dt_fs * units.fs); dyn.run(a.steps)
    for _ in range(a.steps):
        state = ts.nve_step(state, model, dt=a.dt_fs * units.fs)
    dx = np.abs(state.positions.cpu().numpy() - b.positions).max()
    moved = np.abs(b.positions - at.positions).max()
    print(f"E0 torchsim {e_ts:.6f} ase {e_ase:.6f} eV (diff {e_ts - e_ase:.2e}); after {a.steps} steps of {a.dt_fs} fs: "
          f"max |dx| torchsim vs ase {dx:.2e} A (atoms moved up to {moved:.3f} A)")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check"):
        p = sp.add_parser(name); p.add_argument("model"); p.add_argument("xyz")
        if name == "run":
            p.add_argument("out")
        p.add_argument("--charge", type=int, required=True); p.add_argument("--temp", type=float, required=True)
        p.add_argument("--dt-fs", type=float, default=0.5); p.add_argument("--cueq", action="store_true")
        if name == "run":
            p.add_argument("--replicas", type=int, default=32); p.add_argument("--nvt-ps", type=float, default=10)
            p.add_argument("--nve-ps", type=float, default=50); p.add_argument("--save-fs", type=float, default=25)
            p.add_argument("--gamma", type=float, default=10.0); p.add_argument("--seed", type=int, default=0)
            p.add_argument("--float64", action="store_true")
        else:
            p.add_argument("--steps", type=int, default=400)
    p = sp.add_parser("multi"); p.add_argument("model"); p.add_argument("jobs"); p.add_argument("xyz_dir"); p.add_argument("out")
    p.add_argument("--charge", type=int, required=True); p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--cueq", action="store_true"); p.add_argument("--nvt-ps", type=float, default=20)
    p.add_argument("--nve-ps", type=float, default=50); p.add_argument("--save-fs", type=float, default=10)
    p.add_argument("--gamma", type=float, default=10.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--float64", action="store_true"); p.add_argument("--parity", type=int, default=0)
    a = ap.parse_args()
    {"run": run, "check": check, "multi": multi}[a.cmd](a)


if __name__ == "__main__":
    main()
