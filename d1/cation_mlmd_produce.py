# MD producer of the D1 cation finite-T rules (d1/tdep_rules_cation_v1; coordinator 2026-11-26 08:58, decoupled
# protocol): v8 (head q1) MLMD of the species in d1/cation_mlmd_species_v1.csv at each temperature, R replicas each,
# Langevin NVT then NVE, one TorchSim batch per chunk of jobs (container B_torchsim, --cueq). Per job <uid>_T<temp>
# writes <traj dir>/<uid>_T<temp>.npz (uncompressed; written as .tmp.npz and renamed when complete) for the dipole
# consumers (cation_mlmd_dipoles.py):
#   atomic_numbers [N]; positions [R, F, N, 3] float32 (A, every --save-every MD steps of NVE, the first frame one
#   stride after the NVE start); dt_fs (= save-every x MD step, the dipole stride); T_nve [R] (mean NVE temperature,
#   3N-3 degrees of freedom); intact [R]; and for the record: epot, ekin [R, F] float32 (eV), drift_meV_atom [R],
#   first_change_ps [R] (NaN: none), blowup_ps [R] (T > 3 x target; NaN: none), change [R] (str), uid, species, temp,
#   md_dt_fs, nvt_ps, nve_ps, gamma_ps, seed, model.
# intact (as g4_score.py / mace_md_batch.py): no bond-graph change against the minimum (the PAHdb geometry the run
#   starts from; cutoff 1.2 x sum of covalent radii) held over the last 100 fs of NVE, the NVE start graph equal to the
#   minimum's, and no blow-up. Pairs checked: those within 3.5 A in the minimum (a lost bond is always seen; a new bond
#   between atoms further apart needs a lost one first). First change: every 50 fs of NVE (and the NVE start).
#   python cation_mlmd_produce.py run <model> <species csv> <xyz dir> <traj dir> [--temps 300,600,900] [--replicas 8]
#       [--nvt-ps 16] [--nve-ps 64] [--md-dt-fs 0.5] [--save-every 4] [--gamma 10] [--seed 0] [--cueq]
#       [--chunk temp | --chunk N (jobs per chunk)] [--only uid,uid] [--done-flag ALL_MD_DONE] [--gate-uid 5 --gate-dir D]
#   python cation_mlmd_produce.py bench <model> <species csv> <xyz dir> --temps ... [--chunk ...] [--steps 200] [--cueq]
# Chunks run one after another on the GPU; after each chunk a forked CPU process scores and writes its files while the
# GPU runs the next (at most one writer at a time; the parent waits for it before forking the next). Jobs whose .npz
# exists are skipped (restartable per chunk). --done-flag: touched in the traj dir when every job of the run exists.
# --gate-uid U: the 1 fs vs 2 fs smoke gate on identical dynamics. All temperatures of species U go into the first
#   chunk (at its front), and their positions are also kept every save-every/2 MD steps, written to
#   <gate dir>/<U>_T<temp>_fine.npz (same keys, dt_fs halved); the production file's frames are every second fine frame.
import argparse, csv, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_integrity import FACTOR, RCOV, read_xyz, describe
from mace_md_batch import zero_com, per_system

Z = {"H": 1, "C": 6}


def load_jobs(a):
    sp = list(csv.DictReader(open(a.species)))
    if a.only:
        sp = [r for r in sp if r["uid"] in a.only.split(",")]
    temps = [float(t) for t in a.temps.split(",")]
    jobs = [dict(uid=r["uid"], species=r["species"], xyz=f"{a.xyz_dir}/{r['xyz']}", temp=T) for T in temps for r in sp]
    for j in jobs:
        j["id"] = f"{j['uid']}_T{j['temp']:.0f}"
        j["sym"], j["pos"] = read_xyz(j["xyz"])
    return jobs


def chunks(jobs, how, gate=""):
    if gate:                                        # the gate species' jobs first, in the first chunk
        g = [j for j in jobs if j["uid"] == gate]; rest = [j for j in jobs if j["uid"] != gate]
        cs = chunks(rest, how)
        return [g + cs[0]] + cs[1:] if cs else [g]
    if how == "temp":
        out = {}
        for j in jobs:
            out.setdefault(j["temp"], []).append(j)
        return list(out.values())
    n = int(how)
    return [jobs[i:i + n] for i in range(0, len(jobs), n)]


def build(a, jobs, dtype):
    import torch, torch_sim as ts
    from ase import Atoms, units
    from torch_sim.models.mace import MaceModel
    dev = torch.device("cuda")
    ats, temps, meta = [], [], []
    for j in jobs:
        for r in range(a.replicas):
            ats.append(Atoms(j["sym"], positions=j["pos"], cell=np.eye(3) * 100.0, pbc=False)); temps.append(j["temp"])
            meta.append((j, r))
    n_at = sum(len(x) for x in ats)
    if n_at > 30720 and not a.cueq:
        sys.exit(f"{n_at} atoms: use --cueq above 30720 atoms (the plain e3nn path corrupts large batches)")
    model = MaceModel(model=torch.load(a.model, map_location=dev, weights_only=False), device=dev, dtype=dtype,
                      compute_stress=False, enable_cueq=a.cueq, head="q1")
    state = ts.initialize_state(ats, device=dev, dtype=dtype)
    kT = torch.tensor(temps, device=dev, dtype=dtype) * units.kB
    return model, state, kT, meta, n_at


def score_and_write(a, jobs, meta, P, EP, EK, nvt_end_pos, t_ps, fine=None):
    """CPU side (forked child): intact scoring and one file per job."""
    from ase import units
    off = np.concatenate([[0], np.cumsum([len(j["sym"]) for j, _ in meta])])
    F = P.shape[0]; win = max(1, int(round(100 / (a.save_every * a.md_dt_fs)))); coarse = max(1, int(round(50 / (a.save_every * a.md_dt_fs))))
    chk = np.unique(np.concatenate([np.arange(0, F, coarse), np.arange(F - win, F)]))
    for j in jobs:
        idx = [i for i, (jj, _) in enumerate(meta) if jj is j]; N = len(j["sym"]); sym = j["sym"]
        rc = np.array([RCOV[s] for s in sym]); cut = FACTOR * (rc[:, None] + rc[None])
        d0 = np.linalg.norm(j["pos"][:, None] - j["pos"][None], axis=-1)
        ii, jj = np.nonzero(np.triu(d0 < 3.5, 1)); b0 = d0[ii, jj] < cut[ii, jj]; cij = cut[ii, jj]
        pos = np.empty((len(idx), F, N, 3), np.float32); res = dict(T_nve=[], intact=[], drift=[], first=[], blow=[], change=[])
        for k, s in enumerate(idx):
            x = P[:, off[s]:off[s + 1]]; pos[k] = x
            xs = nvt_end_pos[off[s]:off[s + 1]]
            st = np.linalg.norm(xs[ii] - xs[jj], axis=-1) < cij
            b = np.linalg.norm(x[chk][:, ii] - x[chk][:, jj], axis=-1) < cij           # (checked frames, pairs)
            last = np.linalg.norm(x[-win:][:, ii] - x[-win:][:, jj], axis=-1) < cij
            pers = (last != b0).all(0)
            ch = np.nonzero((b != b0).any(1))[0]
            first = np.nan if (st == b0).all() and not len(ch) else (0.0 if not (st == b0).all() else float(t_ps[chk[ch[0]]]))
            Tm = 2 * EK[:, s] / ((3 * N - 3) * units.kB)
            bl = np.nonzero(Tm > 3 * j["temp"])[0]; blow = float(t_ps[bl[0]]) if len(bl) else np.nan
            et = EP[:, s] + EK[:, s]; drift = 1000 * np.polyfit(t_ps, et, 1)[0] * a.nve_ps / N
            gained = {(int(ii[p]), int(jj[p])) for p in np.nonzero(pers & ~b0)[0]}; lost = {(int(ii[p]), int(jj[p])) for p in np.nonzero(pers & b0)[0]}
            res["intact"].append(bool(not pers.any() and (st == b0).all() and np.isnan(blow)))
            res["T_nve"].append(float(Tm.mean())); res["drift"].append(drift); res["first"].append(first); res["blow"].append(blow)
            res["change"].append(describe(sym, gained, lost) if pers.any() else "")
        tmp = f"{a.traj}/{j['id']}.tmp.npz"
        np.savez(tmp, atomic_numbers=np.array([Z[s] for s in sym], np.int64), positions=pos,
                 dt_fs=float(a.save_every * a.md_dt_fs), T_nve=np.array(res["T_nve"]), intact=np.array(res["intact"]),
                 epot=EP[:, idx].T.astype(np.float32), ekin=EK[:, idx].T.astype(np.float32),
                 drift_meV_atom=np.array(res["drift"]), first_change_ps=np.array(res["first"]), blowup_ps=np.array(res["blow"]),
                 change=np.array(res["change"]), uid=j["uid"], species=j["species"], temp=j["temp"], md_dt_fs=a.md_dt_fs,
                 nvt_ps=a.nvt_ps, nve_ps=a.nve_ps, gamma_ps=a.gamma, seed=a.seed, model=a.model)
        os.replace(tmp, f"{a.traj}/{j['id']}.npz")
        if fine is not None and j["uid"] == a.gate_uid:
            fp = np.stack([fine[:, off[s]:off[s + 1]] for s in idx])
            assert np.array_equal(fp[:, 1::2], pos), "fine frames do not interleave the production frames"
            z = dict(np.load(f"{a.traj}/{j['id']}.npz")); z.update(positions=fp, dt_fs=z["dt_fs"] / 2)
            np.savez(f"{a.gate_dir}/{j['id']}_fine.npz", **z)
        print(f"  {j['id']}: intact {sum(res['intact'])}/{len(idx)}, T_nve {np.mean(res['T_nve']):.0f} K, "
              f"|drift| max {np.max(np.abs(res['drift'])):.2f} meV/atom" + "".join(f"; r{k}: {c}" for k, c in enumerate(res["change"]) if c), flush=True)


def run_chunk(a, jobs, seed):
    import torch, torch_sim as ts
    from ase import units
    dtype = torch.float32
    model, state, kT, meta, n_at = build(a, jobs, dtype)
    B = len(meta); dt = a.md_dt_fs * units.fs; gamma = a.gamma / (1000 * units.fs)
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)
    state = ts.nvt_langevin_init(state, model, kT=kT)
    state.momenta = ts.integrators.initialize_momenta(state.positions, state.masses, state.system_idx, kT, gen)
    n_nvt = int(round(a.nvt_ps * 1000 / a.md_dt_fs)); n_nve = int(round(a.nve_ps * 1000 / a.md_dt_fs)); F = n_nve // a.save_every
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n_nvt):
        state = ts.nvt_langevin_step(state, model, dt=dt, kT=kT, gamma=gamma)
    state = zero_com(state); nvt_end = state.positions.float().cpu().numpy()
    P = np.empty((F, n_at, 3), np.float32); EP = np.empty((F, B)); EK = np.empty((F, B))
    ng = sum(len(j["sym"]) for j, _ in meta if j["uid"] == a.gate_uid)    # gate systems are at the front of the batch
    half = a.save_every // 2; FINE = np.empty((2 * F, ng, 3), np.float32) if ng else None; gf = []
    G = 256; gp = torch.empty((G, n_at, 3), device="cuda", dtype=torch.float32)
    ge = torch.empty((G, B), device="cuda", dtype=torch.float64); gk = torch.empty((G, B), device="cuda", dtype=torch.float64)
    m = state.masses; f = 0
    for k in range(1, n_nve + 1):
        state = ts.nve_step(state, model, dt=dt)
        if ng and k % half == 0:
            gf.append(state.positions[:ng].float().clone())
            if len(gf) == 512 or k == n_nve:
                i0 = k // half - len(gf); FINE[i0:i0 + len(gf)] = torch.stack(gf).cpu().numpy(); gf = []
        if k % a.save_every == 0:
            g = f % G
            gp[g] = state.positions; ge[g] = state.energy; gk[g] = per_system(0.5 * (state.momenta ** 2).sum(1) / m, state, B)
            f += 1
            if f % G == 0 or f == F:
                n = (f - 1) % G + 1
                P[f - n:f] = gp[:n].cpu().numpy(); EP[f - n:f] = ge[:n].cpu().numpy(); EK[f - n:f] = gk[:n].cpu().numpy()
    torch.cuda.synchronize(); wall = time.time() - t0
    del model, state, gp, ge, gk; torch.cuda.empty_cache()
    t_ps = np.arange(1, F + 1) * a.save_every * a.md_dt_fs / 1000
    return meta, P, EP, EK, nvt_end, t_ps, wall, n_at, n_nvt + n_nve, FINE


def cmd_run(a):
    os.makedirs(a.traj, exist_ok=True)
    jobs = load_jobs(a); todo = [j for j in jobs if not os.path.exists(f"{a.traj}/{j['id']}.npz")]
    cs = chunks(todo, a.chunk, a.gate_uid if any(j["uid"] == a.gate_uid for j in todo) else ""); child = None
    if a.gate_uid:
        os.makedirs(a.gate_dir, exist_ok=True)
        assert a.save_every % 2 == 0
    print(f"{len(jobs)} jobs, {len(todo)} to run in {len(cs)} chunks", flush=True)
    for c, cj in enumerate(cs):
        meta, P, EP, EK, nvt_end, t_ps, wall, n_at, steps, FINE = run_chunk(a, cj, a.seed + 1000 * c + int(cj[0]["temp"]))
        print(f"chunk {c + 1}/{len(cs)}: {len(cj)} jobs, {len(meta)} systems, {n_at} atoms, {wall:.0f} s, {steps / wall:.1f} steps/s",
              flush=True)
        if child:
            os.waitpid(child, 0)
        sys.stdout.flush(); child = os.fork()
        if child == 0:
            try:
                score_and_write(a, cj, meta, P, EP, EK, nvt_end, t_ps, FINE)
            finally:
                sys.stdout.flush(); os._exit(0)
        del P, EP, EK, FINE
    if child:
        os.waitpid(child, 0)
    if a.done_flag and all(os.path.exists(f"{a.traj}/{j['id']}.npz") for j in jobs):
        open(f"{a.traj}/{a.done_flag}", "w").write(time.strftime("%Y-%m-%d %H:%M:%S %Z\n"))
        print(f"wrote {a.traj}/{a.done_flag}")


def cmd_bench(a):
    import torch, torch_sim as ts
    from ase import units
    jobs = load_jobs(a)
    for c, cj in enumerate(chunks(jobs, a.chunk, a.gate_uid)[:a.max_chunks]):
        model, state, kT, meta, n_at = build(a, cj, torch.float32)
        state = ts.nvt_langevin_init(state, model, kT=kT); dt = a.md_dt_fs * units.fs
        for _ in range(20):
            state = ts.nve_step(state, model, dt=dt)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(a.steps):
            state = ts.nve_step(state, model, dt=dt)
        torch.cuda.synchronize(); w = (time.time() - t0) / a.steps
        print(f"chunk {c}: {len(cj)} jobs, {len(meta)} systems, {n_at} atoms: {1000 * w:.1f} ms/step, "
              f"{n_at / w / 1e6:.2f} M atom-steps/s; 80 ps = {160000 * w / 3600:.2f} h; "
              f"GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        del model, state; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "bench"):
        p = sp.add_parser(name); p.add_argument("model"); p.add_argument("species"); p.add_argument("xyz_dir")
        if name == "run":
            p.add_argument("traj")
        p.add_argument("--temps", default="300,600,900"); p.add_argument("--replicas", type=int, default=8)
        p.add_argument("--nvt-ps", type=float, default=16.0); p.add_argument("--nve-ps", type=float, default=64.0)
        p.add_argument("--md-dt-fs", type=float, default=0.5); p.add_argument("--save-every", type=int, default=4)
        p.add_argument("--gamma", type=float, default=10.0); p.add_argument("--seed", type=int, default=0)
        p.add_argument("--cueq", action="store_true"); p.add_argument("--chunk", default="temp"); p.add_argument("--only", default="")
        p.add_argument("--done-flag", default=""); p.add_argument("--steps", type=int, default=200); p.add_argument("--max-chunks", type=int, default=99)
        p.add_argument("--gate-uid", default=""); p.add_argument("--gate-dir", default="")
    a = ap.parse_args()
    {"run": cmd_run, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
