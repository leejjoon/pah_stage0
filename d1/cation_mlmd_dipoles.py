# Dipole consumer of the D1 cation finite-T rules (d1/tdep_rules_cation_v1; coordinator 2026-11-26 08:58, decoupled
# protocol): polar_v7 (PolarMACE, forward-only) dipoles and Rule B1 IR spectra of the MD trajectories that
# d1/cation_mlmd_produce.py writes to gh200:/var/tmp/jjlee-pah/d1/cation_mlmd/traj/<id>.npz. Every worker (wasp33a
# GPUs 0-3, gh200 GPU 0 once traj/ALL_MD_DONE exists) runs this same script, so all outputs are made the same way.
#   python cation_mlmd_dipoles.py work <stage dir> --dipole polar.model [--remote gh200] [--cache DIR] [--device cuda]
#       [--batch 128] [--gate] [--once] [--delete-traj]
#   python cation_mlmd_dipoles.py one <traj npz> <out _spec.npz> --dipole polar.model [--device cpu] [--frames K]
# Stage dir: traj/, claims/, dipoles/ (on gh200; with --remote HOST, a path on HOST reached over ssh/scp, the npz
# copied to --cache and removed after use). A job is claimed with mkdir claims/<id>.lock (atomic; the lock holds
# host and pid), skipped if dipoles/<id>_spec.npz exists. --gate: the jobs are gate/<id>_fine.npz -> dipoles_gate/.
# Per job: replicas with intact == False dropped (Rule B1); every frame centred at the centre of mass with isotopic
# masses (H 1.00782503, C 12.0: the dipole labels' convention, d1/mlff_dipole_prep.py; a +1 ion's dipole depends on
# the origin); polar_v7 in float32 with compute_force=False on batches of --batch frames (polar_fastgraph.install());
# per replica the Rule B1 spectrum: mlmd_spectra.ir_sigma(dipole, T_nve of that replica, dt_fs of the file, qm=3,
# clip=False) (ACF truncated at FRACTION = 7.987 % of the series: acf_len_s = int(F x FRACTION) x dt_fs).
# Output <id>_spec.npz (written as .tmp.npz, renamed, then re-read as a check):
#   dipole [R_intact, F, 3] float32 (e*A, about the isotopic COM); freq_cm1 [K] (scaled by 0.9578, 300-3800);
#   sigma [R_intact, K] (cm^2, unnormalised); mean [K] (replica mean of sigma / its maximum); replicas [R_intact]
#   (indices into the trajectory's replicas); T_nve [R_intact]; dt_fs; acf_len_s; n_frames; qm=3; clip=False; uid,
#   species, temp, intact [R] (all replicas), model paths.
# --delete-traj removes traj/<id>.npz after the check (the coordinator's disk rule; gh200 keeps them by default).
import argparse, glob, os, socket, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma, FRACTION  # noqa: E402

ISO = {1: 1.00782503, 6: 12.0}


class Dipoles:
    def __init__(self, model, device, batch):
        import mlmd_polar as mp
        self.mp = mp; self.device = device; self.batch = batch
        _, self.dc = mp.calcs(argparse.Namespace(force=None, dipole=model, dtype="float32", charge=1), device)
        self.graphs = {}

    def __call__(self, numbers, P):
        """numbers [N]; P [M, N, 3] float32 -> dipoles [M, 3] float64 (e*A) about the isotopic COM."""
        import torch
        from ase import Atoms
        N = len(numbers); key = tuple(numbers)
        if key not in self.graphs:
            self.graphs = {key: self.mp.Graphs(self.dc, Atoms(numbers=numbers, positions=P[0]), self.batch, 1, 2, self.device)}
        G = self.graphs[key]
        m = torch.tensor([ISO[int(z)] for z in numbers], dtype=torch.float64, device=self.device)
        out = np.empty((len(P), 3))
        with torch.no_grad():
            for i in range(0, len(P), self.batch):
                x = torch.tensor(P[i:i + self.batch], dtype=torch.float64, device=self.device); n = len(x)
                x = x - (m[None, :, None] * x).sum(1, keepdim=True) / m.sum()
                if n < self.batch:                                   # pad the last batch with copies
                    x = torch.cat([x, x[-1:].expand(self.batch - n, N, 3)])
                mu = self.dc.models[0](G.set(x.float()), training=False, compute_force=False)["dipole"]
                out[i:i + n] = mu[:n].double().cpu().numpy()
        return out


def process(dip, src, dst, a, frames=None):
    z = np.load(src); intact = np.asarray(z["intact"], bool); keep = np.nonzero(intact)[0]
    dt = float(z["dt_fs"]); P = z["positions"]; num = z["atomic_numbers"]
    F = P.shape[1] if frames is None else min(frames, P.shape[1])
    mus, sig = [], []; freq = None
    for r in keep:
        mu = dip(num, np.asarray(P[r, :F]))
        f, s = ir_sigma(mu, float(z["T_nve"][r]), dt_fs=dt, qm=3, clip=False)
        freq = f; mus.append(mu.astype(np.float32)); sig.append(s)
    sig = np.array(sig) if sig else np.zeros((0, 0)); mean = sig.mean(0) / sig.mean(0).max() if len(sig) else np.zeros(0)
    tmp = dst[:-4] + ".tmp.npz"
    np.savez(tmp, dipole=np.array(mus).reshape(len(keep), F, 3), freq_cm1=freq if freq is not None else np.zeros(0), sigma=sig,
             mean=mean, replicas=keep, T_nve=np.asarray(z["T_nve"])[keep], dt_fs=dt, acf_len_s=int(F * FRACTION) * dt * 1e-15,
             n_frames=F, qm=3, clip=False, uid=z["uid"], species=z["species"], temp=z["temp"], intact=intact,
             dipole_model=a.dipole, force_model=z["model"], host=socket.gethostname())
    os.replace(tmp, dst)
    c = np.load(dst); assert c["sigma"].shape == sig.shape and np.isfinite(c["sigma"]).all() and float(c["dt_fs"]) == dt
    return len(keep), len(intact), F, dt


def sh(host, cmd):
    return subprocess.run(["ssh", host, cmd], capture_output=True, text=True)


def work(a):
    dip = Dipoles(a.dipole, a.device, a.batch); S = a.stage
    tdir, odir, suf = ("gate", "dipoles_gate", "_fine") if a.gate else ("traj", "dipoles", "")
    me = f"{socket.gethostname()}:{os.getpid()}:{a.device}"
    while True:
        if a.remote:
            r = sh(a.remote, f"cd {S} && ls {tdir} {odir} claims 2>/dev/null; test -e traj/ALL_MD_DONE && echo __DONE__")
            names = r.stdout.split()
        else:
            names = os.listdir(f"{S}/{tdir}") + os.listdir(f"{S}/{odir}") + os.listdir(f"{S}/claims") + \
                (["__DONE__"] if os.path.exists(f"{S}/traj/ALL_MD_DONE") else [])
        ids = sorted(n[:-4] for n in names if n.endswith(f"{suf}.npz") and not n.endswith(".tmp.npz"))
        done = {n[:-9] for n in names if n.endswith("_spec.npz")}; locked = {n[:-5] for n in names if n.endswith(".lock")}
        todo = [i for i in ids if i not in done and i not in locked]
        got = None
        for i in todo:
            ok = sh(a.remote, f"mkdir {S}/claims/{i}.lock && echo {me} > {S}/claims/{i}.lock/owner").returncode == 0 if a.remote else \
                _mkdir(f"{S}/claims/{i}.lock", me)
            if ok:
                got = i; break
        if got is None:
            if a.once or "__DONE__" in names and not todo or a.gate:
                print("no job left", flush=True); return
            time.sleep(60); continue
        t0 = time.time(); src = f"{S}/{tdir}/{got}.npz"; dst = f"{S}/{odir}/{got}_spec.npz"
        if a.remote:
            os.makedirs(a.cache, exist_ok=True); lsrc = f"{a.cache}/{got}.npz"; ldst = f"{a.cache}/{got}_spec.npz"
            subprocess.run(["scp", "-q", f"{a.remote}:{src}", lsrc], check=True)
            nk, nr, F, dt = process(dip, lsrc, ldst, a)
            subprocess.run(["scp", "-q", ldst, f"{a.remote}:{S}/{odir}/{got}_spec.tmp.npz"], check=True)
            r = sh(a.remote, f"cd {S}/{odir} && python3 -c \"import numpy as n; z=n.load('{got}_spec.tmp.npz'); assert n.isfinite(z['sigma']).all()\" "
                             f"&& mv {got}_spec.tmp.npz {got}_spec.npz" + (f" && rm -f {src}" if a.delete_traj else ""))
            assert r.returncode == 0, r.stderr
            os.remove(lsrc); os.remove(ldst)
        else:
            nk, nr, F, dt = process(dip, src, dst, a)
            if a.delete_traj:
                os.remove(src)
        print(f"{got}: {nk}/{nr} intact replicas, {F} frames at {dt} fs, {time.time() - t0:.0f} s ({me})", flush=True)


def _mkdir(p, me):
    try:
        os.mkdir(p)
    except FileExistsError:
        return False
    open(f"{p}/owner", "w").write(me + "\n"); return True


def one(a):
    dip = Dipoles(a.dipole, a.device, a.batch)
    t0 = time.time(); print(process(dip, a.src, a.dst, a, a.frames), f"{time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("work"); w.add_argument("stage"); w.add_argument("--remote", default=""); w.add_argument("--cache", default="/tmp/cation_mlmd_cache")
    w.add_argument("--gate", action="store_true"); w.add_argument("--once", action="store_true"); w.add_argument("--delete-traj", action="store_true")
    o = sp.add_parser("one"); o.add_argument("src"); o.add_argument("dst"); o.add_argument("--frames", type=int, default=None)
    for p in (w, o):
        p.add_argument("--dipole", required=True); p.add_argument("--device", default="cuda"); p.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    {"work": work, "one": one}[a.cmd](a)


if __name__ == "__main__":
    main()
