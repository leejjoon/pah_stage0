# Production viability of a Delta-xTB dipole (GFN2-xTB + MACE correction) along hot cation MD (coordinator plan
# 2026-11-23 21:06, gate (c)): xTB SCF failures, dipole discontinuities (orbital swaps near degeneracy) and wall time
# per frame, against the 800k-1.6M dipole evaluations a production species-temperature needs (1 fs stride).
#   md     python dipole_md_check.py md <force model> <xyz> <out dir> --charge 1 --temp 1200 [--nvt-ps 5 --nve-ps 50]
#          ASE Langevin NVT then NVE (0.5 fs, friction 0.02 per ASE time unit as d1/mlff_accept.py md), COM momentum
#          removed, float32 force model head q<charge>; positions every 1 fs of the NVE part -> traj.npy (A).
#   xtb    python dipole_md_check.py xtb <out dir> --charge 1 [--workers 32]
#          GFN2-xTB along traj.npy in contiguous chunks, each sequential with the previous frame's wavefunction as the
#          guess (as in production); per frame: dipole about the COM (e*A), failure flag, frontier orbital energies
#          (SOMO/HOMO and LUMO), wall time. A failed SCF is retried from scratch and counted. -> xtb.npz
#   mace   python dipole_md_check.py mace <out dir> --charge 1 --dipole name:model_type:model ...  -> mace_<name>.npz
#   mace-batch python dipole_md_check.py mace-batch <out dir> --charge 1 --mult 2 --dipole name:model_type:model
#          [--batch 64] [--stride 1] [--dtype float32]
#          Production mode (coordinator 2026-11-24): MD runs on the force model alone (d1/mace_md_batch.py) and the
#          dipole model is evaluated offline on saved frames, B frames per forward pass, compute_force=False (no
#          autograd graph). ms/frame = wall of the whole loop (graph building included) / frames. -> mace_<name>.npz
#   report python dipole_md_check.py report <out dir> [--dipole-names dxtb_dip,...]
# Jumps: |second difference| of the 1 fs dipole series beyond 10 robust sigma (MAD) and 0.02 D; reported for xTB and
# for each MACE term (which is smooth by construction), with the frontier gap at the flagged frames.
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEBYE_EA = 0.2081943
HARTREE_EV = 27.211386
MASS = {1: 1.00782503, 6: 12.0}


def md(a):
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
    from mlff_accept import calculator, atoms_from
    os.makedirs(a.out, exist_ok=True)
    at = atoms_from(a.xyz); at.calc = calculator(a.force_model, a.charge, a.device, dtype="float32")
    at.set_masses([MASS[z] for z in at.numbers])
    MaxwellBoltzmannDistribution(at, temperature_K=a.temp, rng=np.random.default_rng(a.seed)); Stationary(at); ZeroRotation(at)
    dt = 0.5 * units.fs
    t0 = time.time()
    Langevin(at, dt, temperature_K=a.temp, friction=0.02, rng=np.random.default_rng(a.seed + 1)).run(int(a.nvt_ps * 2000))
    Stationary(at)
    nfr = int(a.nve_ps * 1000)
    traj = np.lib.format.open_memmap(f"{a.out}/traj.npy", mode="w+", dtype=np.float32, shape=(nfr, len(at), 3))
    etot = np.empty(nfr)
    dyn = VelocityVerlet(at, dt)
    for k in range(nfr):
        dyn.run(2)
        traj[k] = at.positions; etot[k] = at.get_potential_energy() + at.get_kinetic_energy()
    traj.flush()
    np.save(f"{a.out}/numbers.npy", at.numbers); np.save(f"{a.out}/etot.npy", etot)
    json.dump(dict(xyz=a.xyz, force_model=a.force_model, charge=a.charge, temp=a.temp, seed=a.seed, nvt_ps=a.nvt_ps,
                   nve_ps=a.nve_ps, wall_s=time.time() - t0, T_nve=float(at.get_temperature()),
                   drift_meV_atom=float((etot[-1] - etot[0]) / len(at) * 1e3)), open(f"{a.out}/md.json", "w"), indent=1)
    print(f"md done: {nfr} frames, {time.time() - t0:.0f} s, NVE drift {(etot[-1] - etot[0]) / len(at) * 1e3:.2f} meV/atom")


def _chunk(args):
    path, lo, hi, num, charge, threads = args
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from tblite.interface import Calculator
    traj = np.load(path, mmap_mode="r")
    m = np.array([MASS[z] for z in num]); B = 0.52917721067
    nel = int(num.sum()) - charge
    mu = np.full((hi - lo, 3), np.nan); fail = np.zeros(hi - lo, int); wall = np.zeros(hi - lo)
    fr = np.full((hi - lo, 3), np.nan)          # occupied-1, highest occupied, lowest unoccupied orbital energy (eV)
    calc = res = None
    for k in range(lo, hi):
        pos = np.asarray(traj[k], dtype=float); t = time.perf_counter()
        for attempt in (0, 1):
            try:
                if calc is None or attempt:
                    calc = Calculator("GFN2-xTB", num, pos / B, charge=charge, uhf=charge % 2); calc.set("verbosity", 0)
                    res = calc.singlepoint()
                else:
                    calc.update(positions=pos / B); res = calc.singlepoint(res)
                break
            except Exception:
                fail[k - lo] += 1; calc = res = None
        wall[k - lo] = time.perf_counter() - t
        if res is None:
            continue
        mu[k - lo] = np.asarray(res.get("dipole")) * B - charge * (m[:, None] * pos).sum(0) / m.sum()
        e = np.asarray(res.get("orbital-energies")) * HARTREE_EV; occ = np.asarray(res.get("orbital-occupations"))
        h = int(np.nonzero(occ > 0.5)[0].max())                 # highest (at least half-)occupied orbital
        fr[k - lo] = (e[h - 1], e[h], e[h + 1])
    return lo, mu, fail, wall, fr


def xtb(a):
    import multiprocessing as mp
    num = np.load(f"{a.out}/numbers.npy"); n = len(np.load(f"{a.out}/traj.npy", mmap_mode="r"))
    if a.frames:
        n = min(n, a.frames)
    edges = np.linspace(0, n, a.workers + 1).astype(int)
    jobs = [(f"{a.out}/traj.npy", lo, hi, num, a.charge, a.threads) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        out = pool.map(_chunk, jobs)
    out.sort(key=lambda x: x[0])
    mu, fail, wall, fr = (np.concatenate([o[i] for o in out]) for i in (1, 2, 3, 4))
    np.savez(f"{a.out}/xtb{a.tag}.npz", mu=mu, fail=fail, wall=wall, frontier=fr, workers=a.workers, threads=a.threads,
             chunk_starts=edges[:-1])
    print(f"xtb: {n} frames, {a.workers} workers x {a.threads} threads, {time.time() - t0:.0f} s; failed SCF attempts "
          f"{int(fail.sum())}; median {np.median(wall) * 1e3:.1f} ms/frame per worker")


def mace(a):
    from mace.calculators import MACECalculator
    from ase import Atoms
    traj = np.load(f"{a.out}/traj.npy", mmap_mode="r"); num = np.load(f"{a.out}/numbers.npy")
    m = np.array([MASS[z] for z in num])
    for spec in a.dipole:
        name, mtype, path = spec.split(":")
        calc = MACECalculator(model_paths=path, device=a.device, default_dtype="float32", model_type=mtype)
        mu = np.empty((len(traj), 3)); wall = np.empty(len(traj))
        for k in range(len(traj)):
            p = np.asarray(traj[k], dtype=float); t = time.perf_counter()
            at = Atoms(num, positions=p - (m[:, None] * p).sum(0) / m.sum()); at.info["charge"] = a.charge; at.calc = calc
            mu[k] = calc.get_property("dipole", at); wall[k] = time.perf_counter() - t
        np.savez(f"{a.out}/mace_{name}.npz", mu=mu, wall=wall)
        print(f"mace {name}: median {np.median(wall) * 1e3:.2f} ms/frame (ASE calculator, one frame per call)")


def mace_batch(a):
    import torch
    from ase import Atoms
    from mace.calculators import MACECalculator
    from mace import data as mace_data
    from mace.tools import torch_geometric
    traj = np.load(f"{a.out}/traj.npy", mmap_mode="r"); num = np.load(f"{a.out}/numbers.npy")
    m = np.array([MASS[z] for z in num]); idx = np.arange(0, len(traj), a.stride)
    if not a.slow_graph:
        import polar_fastgraph; polar_fastgraph.install()      # vectorised PolarMACE all-pairs graph (no-op otherwise)
    for spec in a.dipole:
        name, mtype, path = spec.split(":")
        calc = MACECalculator(model_paths=path, device=a.device, default_dtype=a.dtype, model_type=mtype)
        model = calc.models[0]; keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
        mu = np.empty((len(idx), 3)); torch.cuda.synchronize(); t0 = time.perf_counter(); tg = 0.0
        for s0 in range(0, len(idx), a.batch):
            tb = time.perf_counter(); graphs = []
            for k in idx[s0:s0 + a.batch]:
                p = np.asarray(traj[k], dtype=float)
                at = Atoms(num, positions=p - (m[:, None] * p).sum(0) / m.sum()); at.info.update(charge=a.charge, spin=a.mult)
                cfg = mace_data.config_from_atoms(at, key_specification=keyspec, head_name=calc.head)
                graphs.append(mace_data.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads))
            batch = next(iter(torch_geometric.dataloader.DataLoader(graphs, batch_size=len(graphs), shuffle=False))).to(a.device)
            tg += time.perf_counter() - tb
            with torch.no_grad():
                out = model(batch.to_dict(), training=False, compute_force=False)
            mu[s0:s0 + len(graphs)] = out["dipole"].detach().cpu().numpy()
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        np.savez(f"{a.out}/mace_{name}.npz", mu=mu, wall=np.full(len(idx), wall / len(idx)), stride=a.stride, batch=a.batch,
                 graph_s=tg)
        print(f"mace-batch {name}: B={a.batch} {len(idx)} frames {wall:.1f} s -> {wall / len(idx) * 1e3:.2f} ms/frame "
              f"(graph building {tg / len(idx) * 1e3:.2f} ms/frame)")


def jumps(mu_D):
    """Second differences of a 1 fs dipole series (D); a from-scratch SCF at an xTB chunk start that lands on another
    solution shows up here like any other jump."""
    d2 = np.full(len(mu_D), np.nan)
    d2[1:-1] = np.linalg.norm(mu_D[2:] - 2 * mu_D[1:-1] + mu_D[:-2], axis=1)
    med = np.nanmedian(d2); sig = 1.4826 * np.nanmedian(np.abs(d2 - med))
    thr = max(med + 10 * sig, 0.02)
    return d2, thr, np.nonzero(d2 > thr)[0]


def report(a):
    md_info = json.load(open(f"{a.out}/md.json")); x = np.load(f"{a.out}/xtb{a.tag}.npz")
    mu = x["mu"] / DEBYE_EA; fr = x["frontier"]
    gap_lu = fr[:, 2] - fr[:, 1]; gap_hh = fr[:, 1] - fr[:, 0]
    d2, thr, J = jumps(mu)
    out = dict(md=md_info, frames=len(mu), scf_failed_attempts=int(x["fail"].sum()), frames_without_dipole=int(np.isnan(mu[:, 0]).sum()),
               ms_per_frame_median=float(np.median(x["wall"]) * 1e3), ms_per_frame_p99=float(np.percentile(x["wall"], 99) * 1e3),
               workers=int(x["workers"]), threads=int(x["threads"]),
               xtb_jump_threshold_D=float(thr), xtb_jumps=len(J), xtb_d2_median_D=float(np.nanmedian(d2)), xtb_d2_max_D=float(np.nanmax(d2)),
               gap_somo_lumo_eV_min=float(np.nanmin(gap_lu)), gap_somo_below_eV_min=float(np.nanmin(gap_hh)),
               jump_frames=[dict(frame=int(k), d2_D=float(d2[k]), gap_lumo_eV=float(gap_lu[k]), gap_below_eV=float(gap_hh[k])) for k in J[:50]])
    for name in a.dipole_names.split(",") if a.dipole_names else []:
        y = np.load(f"{a.out}/mace_{name}.npz"); d2m, thrm, Jm = jumps(y["mu"] / DEBYE_EA)
        tot = mu + y["mu"] / DEBYE_EA; d2t, thrt, Jt = jumps(tot)
        out[f"mace_{name}"] = dict(ms_per_frame_median=float(np.median(y["wall"]) * 1e3), d2_median_D=float(np.nanmedian(d2m)),
                                   d2_max_D=float(np.nanmax(d2m)), jumps=len(Jm), total_jumps=len(Jt), total_d2_max_D=float(np.nanmax(d2t)))
    json.dump(out, open(f"{a.out}/report{a.tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "jump_frames"}, indent=1))
    for j in out["jump_frames"][:10]:
        print("  jump", j)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("md")
    for k in ("force_model", "xyz", "out"):
        p.add_argument(k)
    p.add_argument("--charge", type=int, default=1); p.add_argument("--temp", type=float, default=1200)
    p.add_argument("--nvt-ps", type=float, default=5); p.add_argument("--nve-ps", type=float, default=50)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda")
    p = sub.add_parser("xtb"); p.add_argument("out"); p.add_argument("--charge", type=int, default=1)
    p.add_argument("--workers", type=int, default=32); p.add_argument("--threads", type=int, default=1)
    p.add_argument("--frames", type=int, default=0); p.add_argument("--tag", default="")
    p = sub.add_parser("mace"); p.add_argument("out"); p.add_argument("--charge", type=int, default=1)
    p.add_argument("--dipole", action="append", default=[]); p.add_argument("--device", default="cuda")
    p = sub.add_parser("mace-batch"); p.add_argument("out"); p.add_argument("--charge", type=int, default=1)
    p.add_argument("--mult", type=int, default=2); p.add_argument("--dipole", action="append", default=[])
    p.add_argument("--batch", type=int, default=64); p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dtype", default="float32"); p.add_argument("--device", default="cuda")
    p.add_argument("--slow-graph", action="store_true", help="keep graph_longrange's looped all-pairs graph")
    p = sub.add_parser("report"); p.add_argument("out"); p.add_argument("--dipole-names", default=""); p.add_argument("--tag", default="")
    a = ap.parse_args()
    dict(md=md, xtb=xtb, mace=mace, report=report)[a.cmd](a) if a.cmd != "mace-batch" else mace_batch(a)


if __name__ == "__main__":
    main()
