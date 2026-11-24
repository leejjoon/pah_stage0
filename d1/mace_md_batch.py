# Batched MACE MLMD for D1 production (course correction 2026-11-23, plan "Status log"): one species, charge and
# temperature per call, B independent replicas on one GPU through TorchSim (container B_torchsim), instead of serial
# single-molecule ASE runs (mlff_accept.py md, ~95 steps/s/GPU).
#   python mace_md_batch.py run <model> <minimum.xyz> <out prefix> --charge Q --temp T [--replicas 32] [--nvt-ps 10]
#                               [--nve-ps 50] [--save-fs 25] [--dt-fs 0.5] [--gamma 10] [--seed 0] [--cueq] [--float64]
#   python mace_md_batch.py check <model> <minimum.xyz> --charge Q --temp T [--steps 400]
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
    a = ap.parse_args()
    run(a) if a.cmd == "run" else check(a)


if __name__ == "__main__":
    main()
