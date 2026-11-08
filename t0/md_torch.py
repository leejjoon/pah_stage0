"""T0 PyTorch driver (plan 4.3): MACE-OFF small on cuda, float32, no TF32. Run in container B (B_torchsim for
--mode torchsim), pinned with CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0.

Modes ("batch b" = b trajectories of total work, trajectory k = species k % 20, as in md_jax.py):
  sequential  ASE Atoms per molecule, ase.md.langevin.Langevin (fixcm=False), one after another in one process
  mps         P worker processes (--workers) take the b trajectories from a shared queue; run under a CUDA MPS daemon
              started by t0/run_mps.sh on the host (asserts that exactly b jobs completed)
  torchsim    all b molecules in one TorchSim SimState, MaceTorchSimModel, nvt_langevin/nve
  concat      all b molecules in ONE ASE Atoms, >= 30 A apart, one MACE call per step; min separation asserted
Common: identical initial velocities from t0/common.py, no COM removal or constraint; stress off everywhere
(equal work: energy, forces and the dipole proxy); per-step temperature, energy and dipole proxy recorded.
"""
import argparse, csv, importlib.metadata, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["sequential", "mps", "torchsim", "concat"])
ap.add_argument("--cueq", action="store_true")
ap.add_argument("--xyz", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "species_T0.xyz"))
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--equil", type=int, default=C.EQUIL_STEPS)
ap.add_argument("--prod", type=int, default=C.PROD_STEPS)
ap.add_argument("--ensemble", default="nvt", choices=["nvt", "nve"])
ap.add_argument("--workers", type=int, default=4, help="MPS pool size P")
ap.add_argument("--threads", type=int, default=1, help="CPU threads per process (OMP_NUM_THREADS, torch.set_num_threads)")
ap.add_argument("--save-vel", type=int, default=0)
ap.add_argument("--chunk", type=int, default=1000, help="torchsim: observables copied to host every chunk steps")
ap.add_argument("--out", default="md_torch")
ap.add_argument("--csv", default=None)
ap.add_argument("--repeat", type=int, default=0)
ap.add_argument("--seed", type=int, default=20261108)
args = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(args.threads)


def setup_torch():
    import torch
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" not in os.environ, "unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE (NGC sets it)"
    torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_num_threads(args.threads)
    return torch


def ase_calc(torch):
    from mace.calculators import mace_off
    calc = mace_off(model="small", default_dtype="float32", device="cuda", enable_cueq=args.cueq)
    calc.compute_stress = False                       # equal work: the calculator computes stress by default
    return calc


def ase_atoms(Z, X, V):
    from ase import Atoms
    a = Atoms(numbers=Z, positions=X, masses=[C.MASS[z] for z in Z])
    a.set_velocities(V * C.FS_PER_TU)                  # Angstrom/fs -> ASE velocity units (Angstrom per 10.18 fs)
    return a


def ase_dynamics(a):
    from ase import units
    if args.ensemble == "nvt":
        from ase.md.langevin import Langevin
        return Langevin(a, timestep=C.DT_FS * units.fs, temperature_K=C.T_K, friction=C.GAMMA_PER_FS / units.fs,
                        fixcm=False, rng=np.random.default_rng([args.seed, 1, int(a.info.get("traj", 0))]))
    from ase.md.verlet import VelocityVerlet
    return VelocityVerlet(a, timestep=C.DT_FS * units.fs)


def run_ase_traj(k, Z, X, calc, torch, save_vel=False, nsteps=None):
    """One trajectory (sequential and MPS modes). Returns observables and the production wall time.
    nsteps = (equil, prod) overrides the command line (used for the untimed MPS warm-up)."""
    equil, prod = nsteps or (args.equil, args.prod)
    a = ase_atoms(Z, X, C.initial_velocities(Z, k, args.seed)); a.info["traj"] = k; a.calc = calc
    q = C.proxy_charges(Z, X); m = a.get_masses()
    dyn = ase_dynamics(a)
    obs = {"T": [], "epot": [], "eint": [], "ke": [], "dip": [], "vel": []}
    def record():
        v = a.get_velocities() / C.FS_PER_TU
        obs["T"].append(C.temperature(v, m)); obs["epot"].append(a.get_potential_energy()); obs["ke"].append(a.get_kinetic_energy())
        obs["eint"].append(np.asarray(calc.results["node_energy"], np.float64).sum())   # E0 already subtracted
        obs["dip"].append(q @ a.positions)
        if save_vel: obs["vel"].append(v.astype(np.float32))
    if equil: dyn.run(equil)
    dyn.attach(record, interval=1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    dyn.run(prod)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    return {k_: np.array(v) for k_, v in obs.items()}, wall


def mps_worker(wid, jobs, results, start, mols):
    import faulthandler, signal; faulthandler.register(signal.SIGUSR1, all_threads=True)   # kill -USR1 <pid> dumps stacks
    torch = setup_torch(); calc = ase_calc(torch)
    Z, X = mols[0]; run_ase_traj(0, Z, X, calc, torch, nsteps=(0, 5))   # warm-up (CUDA context, kernels), untimed
    results.put(("ready", wid, 0, os.getpid())); start.wait()             # timing starts when every worker is ready
    done = 0
    while True:
        k = jobs.get()        # blocking: get_nowait() raises Empty spuriously while another worker holds the read lock
        if k is None: break   # one sentinel per worker
        Z, X = mols[k]; o, wall = run_ase_traj(k, Z, X, calc, torch); done += 1
        results.put((k, wid, wall, float(o["T"].mean())))
    results.put(("done", wid, done, os.getpid()))


def write_csv(row):
    if not args.csv: return
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)


def report(wall, b, mode, impl, workers="", extra=None, peak=0.0):
    ps = args.prod * C.DT_FS / 1000
    print(f"{mode}{'/' + impl if impl else ''}: production {args.prod} steps ({ps:g} ps) x {b} trajectories: {wall:.1f} s wall -> "
          f"{wall / (ps * b):.3f} s per molecule-ps, {86400 / (wall / ps) / 1000:.3f} ns/day per trajectory")
    import torch
    write_csv(dict(stack="torch", mode=mode, batched_impl=impl, cueq="yes" if args.cueq else "no",
                   cueq_version=importlib.metadata.version("cuequivariance"), conv_fusion="", dtype="float32",
                   matmul_precision=torch.get_float32_matmul_precision(), batch=b, mps_workers=workers, repeat=args.repeat,
                   species_set="T0_20", ensemble=args.ensemble, wall_s=round(wall, 3), s_per_mol_ps=round(wall / (ps * b), 5),
                   ns_per_day_per_traj=round(86400 / (wall / ps) / 1000, 5), gpu_util="", gpu_mem_gb=round(peak, 3), compile_s="",
                   steps=args.prod, edge_cap="", cuex="", **(extra or {})))


def main():
    species = C.read_xyz(args.xyz)
    mols = C.batch(species, args.batch); b = len(mols)
    nat = np.array([len(z) for z, _ in mols])

    if args.mode == "sequential":
        torch = setup_torch(); calc = ase_calc(torch)
        print("calculator compute_stress:", calc.compute_stress, "| matmul precision:", torch.get_float32_matmul_precision())
        walls, Ts, out = [], [], {}
        for k, (Z, X) in enumerate(mols):
            o, wall = run_ase_traj(k, Z, X, calc, torch, save_vel=k < args.save_vel)
            walls.append(wall); Ts.append(o["T"].mean())
            out.update({f"{key}_{k}": v for key, v in o.items() if len(v)})
        print(f"T mean {np.mean(Ts):.1f} K (per-trajectory {min(Ts):.0f}-{max(Ts):.0f} K)")
        if args.ensemble == "nve":
            drift = [abs((out[f"eint_{k}"][-1] + out[f"ke_{k}"][-1]) - (out[f"eint_{k}"][0] + out[f"ke_{k}"][0])) / nat[k] for k in range(b)]
            print(f"NVE drift of interaction + kinetic energy: max |dE|/atom {max(drift):.2e} eV over {args.prod * C.DT_FS / 1000:g} ps")
        np.savez_compressed(args.out + ".npz", nat=nat, **out)
        report(sum(walls), b, "sequential", "", peak=torch.cuda.max_memory_allocated() / 1e9)

    elif args.mode == "mps":
        import multiprocessing as mp
        pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        assert pipe and os.path.exists(os.path.join(pipe, "control")), f"no MPS control pipe at {pipe}; start the daemon with t0/run_mps.sh"
        ctx = mp.get_context("spawn"); jobs, results = ctx.Queue(), ctx.Queue()
        P = min(args.workers, b)
        for k in list(range(b)) + [None] * P: jobs.put(k)
        start = ctx.Event()
        procs = [ctx.Process(target=mps_worker, args=(w, jobs, results, start, mols)) for w in range(P)]
        [p.start() for p in procs]
        for _ in range(P):                     # model loading and warm-up are not timed
            r = results.get(timeout=1800); assert r[0] == "ready", r
        t0 = time.perf_counter(); start.set()
        finished, per_worker, pids, Ts, walls = 0, {}, [], [], []
        while finished < P:
            r = results.get(timeout=36000)
            if r[0] == "done": finished += 1; per_worker[r[1]] = r[2]; pids.append(r[3])
            else: Ts.append(r[3]); walls.append(r[2])
        [p.join() for p in procs]; wall = time.perf_counter() - t0
        assert sum(per_worker.values()) == b, f"{sum(per_worker.values())} jobs completed, expected {b}"
        print(f"MPS pool P={P}: jobs per worker {sorted(per_worker.values())}, T mean {np.mean(Ts):.1f} K; worker pids {sorted(pids)}")
        print(f"per-trajectory production wall: {np.min(walls):.2f}-{np.max(walls):.2f} s (sum {np.sum(walls):.1f} s over {P} workers)")
        print("note: wall time from the start barrier (models loaded, warmed up) to the last worker finishing; it includes equilibration")
        report(wall, b, "mps_pool", "", workers=P, extra=None)

    elif args.mode == "torchsim":
        torch = setup_torch()
        import torch_sim as ts
        from torch_sim.integrators import nvt_langevin_init, nvt_langevin_step, nve_init, nve_step
        from torch_sim.units import MetalUnits as MU
        from mace.calculators import mace_off
        from mace.calculators.mace_torchsim import MaceTorchSimModel
        dev = torch.device("cuda")
        raw = mace_off(model="small", default_dtype="float32", device="cuda", return_raw_model=True)
        model = MaceTorchSimModel(model=raw, device=dev, dtype=torch.float32, compute_forces=True, compute_stress=False, enable_cueq=args.cueq)
        ncueq = sum(type(mm).__module__.startswith("cuequivariance") for mm in model.model.modules()) if hasattr(model, "model") else "?"
        print(f"TorchSim model dtype {model.dtype}, stress {model.compute_stress}, cuequivariance modules {ncueq}")
        atoms = [ase_atoms(Z, X, C.initial_velocities(Z, k, args.seed)) for k, (Z, X) in enumerate(mols)]
        state = ts.io.atoms_to_state(atoms, device=dev, dtype=torch.float32); state.rng = args.seed
        V0 = np.concatenate([C.initial_velocities(Z, k, args.seed) for k, (Z, X) in enumerate(mols)])
        m = torch.tensor(np.concatenate([[C.MASS[z] for z in Z] for Z, _ in mols]), device=dev, dtype=torch.float32)
        assert torch.allclose(state.masses, m), "TorchSim masses differ from t0/common.py"
        state.momenta = m[:, None] * torch.tensor(V0 * C.FS_PER_TU, device=dev, dtype=torch.float32)   # same internal unit (10.18 fs)
        kT = torch.tensor(C.T_K * MU.temperature, device=dev, dtype=torch.float32)
        dt = torch.tensor(C.DT_FS * 1e-3 * MU.time, device=dev, dtype=torch.float32)
        gamma = torch.tensor(C.GAMMA_PER_FS * 1e3 / MU.time, device=dev, dtype=torch.float32)
        print(f"internal: kT {kT.item():.5f} eV, dt {dt.item():.5f}, gamma {gamma.item():.5f}, gamma*dt {(gamma * dt).item():.5f} (expect 0.005)")
        init, stepf = (nvt_langevin_init, lambda s: nvt_langevin_step(s, model, dt=dt, kT=kT, gamma=gamma)) if args.ensemble == "nvt" \
            else (nve_init, lambda s: nve_step(s, model, dt=dt))
        state = init(state, model, kT=kT)
        assert torch.allclose(state.momenta, m[:, None] * torch.tensor(V0 * C.FS_PER_TU, device=dev, dtype=torch.float32)), "momenta were re-drawn"
        sid = state.system_idx; q = torch.tensor(np.concatenate([C.proxy_charges(Z, X) for Z, X in mols]), device=dev, dtype=torch.float32)
        dof = torch.tensor(3 * nat, device=dev, dtype=torch.float32)
        nsave = int(nat[:args.save_vel].sum())
        def observe(s):
            ke = torch.zeros(b, device=dev).index_add_(0, sid, 0.5 * (s.momenta ** 2 / s.masses[:, None]).sum(1))
            dip = torch.zeros(b, 3, device=dev).index_add_(0, sid, q[:, None] * s.positions)
            return 2 * ke / (dof * C.KB), s.energy.clone(), dip
        for _ in range(args.equil): state = stepf(state)
        obs, vel = [], []
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
        buf = []
        for i in range(args.prod):
            state = stepf(state); buf.append(observe(state))
            if nsave: vel.append((state.momenta[:nsave] / state.masses[:nsave, None] / C.FS_PER_TU).clone())
            if len(buf) == args.chunk or i == args.prod - 1:
                obs += [tuple(t.cpu().numpy() for t in o) for o in buf]; buf = []
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        T = np.array([o[0] for o in obs]); E = np.array([o[1] for o in obs]); D = np.array([o[2] for o in obs])
        print(f"T mean {T.mean():.1f} K (per-trajectory mean {T.mean(0).min():.0f}-{T.mean(0).max():.0f} K)")
        np.savez_compressed(args.out + ".npz", T=T, epot=E, dip=D, nat=nat, **({"vel": torch.stack(vel).cpu().numpy()} if nsave else {}))
        report(wall, b, "batched", "torchsim", peak=torch.cuda.max_memory_allocated() / 1e9)

    elif args.mode == "concat":
        torch = setup_torch(); calc = ase_calc(torch)
        from ase import Atoms
        RAD = max(np.linalg.norm(X - X.mean(0), axis=1).max() for _, X in mols)
        SEP = 30.0 + 2 * RAD                               # >= 30 A between molecules (plan 4.3), not between centres
        nx = int(np.ceil(b ** (1 / 3)))
        grid = [np.array([i, j, k]) * SEP for i in range(nx) for j in range(nx) for k in range(nx)][:b]
        Zs, Xs, Vs, mol = [], [], [], []
        for k, (Z, X) in enumerate(mols):
            Zs.append(Z); Xs.append(X - X.mean(0) + grid[k]); Vs.append(C.initial_velocities(Z, k, args.seed)); mol.append(np.full(len(Z), k))
        a = ase_atoms(np.concatenate(Zs), np.concatenate(Xs), np.concatenate(Vs)); a.calc = calc
        mol = np.concatenate(mol); q = np.concatenate([C.proxy_charges(Z, X) for Z, X in mols]); m = a.get_masses()
        rcut = float(calc.r_max) if hasattr(calc, "r_max") else 4.5
        def min_sep(pos):
            cen = np.array([pos[mol == k].mean(0) for k in range(b)])
            rad = np.array([np.linalg.norm(pos[mol == k] - cen[k], axis=1).max() for k in range(b)])
            if b == 1: return np.inf
            d = np.linalg.norm(cen[:, None] - cen[None], axis=-1) - rad[:, None] - rad[None]
            return d[~np.eye(b, dtype=bool)].min()
        print(f"concatenated Atoms: {len(a)} atoms, molecules on a {SEP:g} A grid; initial min separation {min_sep(a.positions):.1f} A (cutoff {rcut} A)")
        dyn = ase_dynamics(a)
        Ts, E, D = [], [], []
        def record():
            v = a.get_velocities() / C.FS_PER_TU
            ke = np.bincount(mol, 0.5 * (m[:, None] * v ** 2).sum(1) * C.AMU_A2_PER_FS2_IN_EV, minlength=b)
            Ts.append(2 * ke / (3 * nat * C.KB)); E.append(a.get_potential_energy())
            D.append(np.array([q[mol == k] @ a.positions[mol == k] for k in range(b)]))
        if args.equil: dyn.run(args.equil)
        dyn.attach(record, interval=1)
        torch.cuda.synchronize(); t0 = time.perf_counter(); dyn.run(args.prod); torch.cuda.synchronize(); wall = time.perf_counter() - t0
        ms = min_sep(a.positions)
        assert ms > 2 * rcut, f"molecules came within {ms:.1f} A (< 2 x cutoff)"
        T = np.array(Ts)
        print(f"T mean {T.mean():.1f} K (per-trajectory mean {T.mean(0).min():.0f}-{T.mean(0).max():.0f} K); final min separation {ms:.1f} A")
        np.savez_compressed(args.out + ".npz", T=T, epot=np.array(E), dip=np.array(D), nat=nat)
        report(wall, b, "batched", "concat", peak=torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":   # spawned MPS workers import this module; only the parent runs main()
    main()
