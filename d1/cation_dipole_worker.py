# Dipole consumer for the decoupled tdep_rules_cation_v1 campaign (coordinator 2026-11-26 08:58 KST): gh200 GPU 0 streams
# v8 cation MD coordinates into <root>/traj/<uid>_T<temp>.npz; each worker (one per GPU, on wasp33a or on gh200 once MD
# ends) claims a file, evaluates the frozen polar_v7 dipole model on every frame and uploads the Committee Rule B1 IR
# spectrum to <root>/dipoles/<id>_spec.npz. Claims are atomic: mkdir <root>/claims/<id>.lock (on the root host).
#
#   python cation_dipole_worker.py --dipole polar.model [--host gh200 | --host local] [--root /var/tmp/jjlee-pah/d1/cation_mlmd]
#       [--scratch ~/pah/cation_dipole/scratch] [--batch 128] [--poll 60] [--max-jobs 0] [--only <id>] [--keep]
#   (one process per GPU: CUDA_VISIBLE_DEVICES=k)
#
# Input traj npz (producer, mace_md_batch.py multi): atomic_numbers [N], positions [R, F, N, 3] float32 (A), dt_fs (2.0),
# T_nve [R] (measured NVE mean T), intact [R] (bool, g4_score persistence). Writers use <id>.tmp.npz then mv.
# Per job:
#   1. claim (mkdir), pull traj/<id>.npz to local scratch (skipped with --host local: read in place);
#   2. keep replicas with intact == True (Committee Rule B1);
#   3. COM-centre every frame with isotopic masses (g16_io.MASS, the mlff_dipole_prep.py / dft_dipole label convention, so
#      the Q = +1 origin term is zero) and evaluate polar_v7 (polar_fastgraph.install(), float32, compute_force=False,
#      charge 1, mult 2) in batches of B frames;
#   4. Rule B1 spectrum per intact replica: mlmd_spectra.ir_sigma(mu, T_nve[r], dt_fs, qm=3, clip=False);
#   5. write dipoles/<id>_spec.tmp.npz on the root host, check it loads there (keys, shapes, finite), mv to
#      dipoles/<id>_spec.npz; then delete the local copy and traj/<id>.npz on the root host. The lock directory stays
#      (holds claim.txt and done.txt).
# Output keys: profile_fit.py's (freq, sigma [R_intact, F] cm^2 unnormalised, mean (replica mean, peak 1), replicas,
# prefactor_T, qm, clip) plus freq_cm1 (= freq), sigma_mean (unnormalised replica mean), dipole [R_intact, F_md, 3]
# float32 (e*A, about the COM), T_nve, intact (all R), n_intact, dt_fs, T (nominal), uid, atomic_numbers, dipole_model,
# worker. If no replica is intact, sigma is (0, F) and mean is zeros.
# Exits when <root>/traj/ALL_MD_DONE exists and no unclaimed traj/*.npz remain. A worker that dies leaves its lock with
# no done.txt; such ids are listed at exit and need a manual rmdir to be redone.
import argparse, json, os, random, shlex, socket, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma  # noqa: E402
from g16_io import MASS  # noqa: E402

ISO = {1: MASS["H"], 6: MASS["C"]}
QM, CLIP = 3, False

VERIFY = r"""
import sys, numpy as np
z = np.load(sys.argv[1]); n = int(z["n_intact"])
assert z["sigma"].shape == (n, len(z["freq"])) and z["dipole"].shape[0] == n and len(z["T_nve"]) == n
assert np.isfinite(z["sigma"]).all() and np.isfinite(z["dipole"]).all() and float(z["dt_fs"]) > 0
print("ok", n, z["dipole"].shape, len(z["freq"]))
"""


class Root:
    """Shell and file access to the staging root, over ssh (--host gh200) or in place (--host local)."""

    def __init__(self, host, root):
        self.host, self.root = host, root.rstrip("/")

    def sh(self, cmd, check=True):
        args = ["bash", "-c", cmd] if self.host == "local" else ["ssh", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "VisualHostKey=no", self.host, cmd]
        for attempt in range(5):
            p = subprocess.run(args, capture_output=True, text=True)
            if p.returncode != 255 or self.host == "local":       # 255 = ssh connection failure: retry
                break
            time.sleep(10 * (attempt + 1))
        if check and p.returncode != 0:
            raise RuntimeError(f"{cmd!r} -> {p.returncode}: {p.stderr.strip()}")
        return p

    def path(self, rel):
        return f"{self.root}/{rel}"

    def get(self, rel, dst):
        if self.host == "local":
            return self.path(rel)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "VisualHostKey=no", f"{self.host}:{self.path(rel)}", dst], check=True)
        return dst

    def put(self, src, rel):
        if self.host == "local":
            subprocess.run(["cp", src, self.path(rel)], check=True)
        else:
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "VisualHostKey=no", src, f"{self.host}:{self.path(rel)}"], check=True)

    def listing(self):
        """(traj ids ready, claimed ids, ALL_MD_DONE present)"""
        q = shlex.quote
        out = self.sh(f"cd {q(self.root)} && ls traj claims; test -e traj/ALL_MD_DONE && echo __ALL_MD_DONE__", check=False).stdout
        ready, claimed, sect = set(), set(), None
        for ln in out.splitlines():
            if ln.endswith(":"):
                sect = ln[:-1]; continue
            if ln == "__ALL_MD_DONE__" or not ln:
                continue
            if sect == "traj" and ln.endswith(".npz") and not ln.endswith(".tmp.npz"):
                ready.add(ln[:-4])
            elif sect == "claims" and ln.endswith(".lock"):
                claimed.add(ln[:-5])
        return ready, claimed, "__ALL_MD_DONE__" in out

    def claim(self, jid, note):
        p = self.sh(f"mkdir {shlex.quote(self.path(f'claims/{jid}.lock'))}", check=False)
        if p.returncode != 0:
            return False
        self.sh(f"echo {shlex.quote(note)} > {shlex.quote(self.path(f'claims/{jid}.lock/claim.txt'))}", check=False)
        return True


class Dipoles:
    """polar_v7 forward-only on batches of frames of one molecule."""

    def __init__(self, model, numbers, B, device="cuda"):
        import torch
        from ase import Atoms
        from mace.calculators import MACECalculator
        import polar_fastgraph; polar_fastgraph.install()
        from mlmd_polar import Graphs
        self.torch, self.B = torch, B
        dc = MACECalculator(model_paths=model, device=device, default_dtype="float32", model_type="PolarMACE")
        at = Atoms(numbers=numbers, positions=np.zeros((len(numbers), 3)))
        self.G = Graphs(dc, at, B, 1, 2, device)          # charge 1, mult 2: info charge/spin -> model inputs
        self.model, self.device = dc.models[0], device
        self.m = torch.tensor([ISO[int(z)] for z in numbers], dtype=torch.float64, device=device)

    def __call__(self, pos):
        """pos: (F, N, 3) float32 numpy -> (F, 3) float64 e*A about the isotopic COM"""
        torch, B = self.torch, self.B
        F = len(pos); out = np.empty((F, 3))
        for s in range(0, F, B):
            x = torch.as_tensor(pos[s:s + B], device=self.device).double()
            n = len(x)
            if n < B:                                       # pad the last batch with copies of its first frame
                x = torch.cat([x, x[:1].expand(B - n, -1, -1)])
            x = x - (self.m[None, :, None] * x).sum(1, keepdim=True) / self.m.sum()
            with torch.no_grad():
                mu = self.model(self.G.set(x), training=False, compute_force=False)["dipole"]
            out[s:s + n] = mu.reshape(B, 3)[:n].double().cpu().numpy()
        return out


def process(a, R, jid, ev_cache, log):
    t0 = time.time()
    os.makedirs(a.scratch, exist_ok=True)
    local = R.get(f"traj/{jid}.npz", f"{a.scratch}/{jid}.npz")
    t_pull = time.time() - t0
    z = np.load(local)
    num = np.asarray(z["atomic_numbers"]).astype(int)
    assert set(num.tolist()) <= {1, 6}, f"{jid}: non C/H atoms"
    intact = np.asarray(z["intact"]).astype(bool); T_nve = np.asarray(z["T_nve"], float); dt_fs = float(z["dt_fs"])
    P = z["positions"]                                       # [R, F, N, 3]
    n_frames = P.shape[1]
    assert P.ndim == 4 and P.shape[0] == len(intact) == len(T_nve) and P.shape[2] == len(num), (P.shape, intact.shape)
    key = tuple(num.tolist())
    if key not in ev_cache:
        ev_cache.clear(); ev_cache[key] = Dipoles(a.dipole, num, a.batch)
    ev = ev_cache[key]
    t1 = time.time()
    reps = np.flatnonzero(intact)
    mu = np.stack([ev(np.asarray(P[r], np.float32)) for r in reps]) if len(reps) else np.zeros((0, n_frames, 3))
    t_eval = time.time() - t1
    del P, z
    spec = [ir_sigma(mu[k], T_nve[r], dt_fs, qm=QM, clip=CLIP) for k, r in enumerate(reps)]
    if spec:
        freq = spec[0][0]; sig = np.array([s[1] for s in spec])
    else:
        freq = ir_sigma(np.zeros((n_frames, 3)), 300.0, dt_fs, qm=QM, clip=CLIP)[0]
        sig = np.zeros((0, len(freq)))
    smean = sig.mean(0) if len(sig) else np.zeros(len(freq))
    mean = smean / smean.max() if len(sig) and smean.max() > 0 else smean
    uid, _, temp = jid.rpartition("_T")
    out = f"{a.scratch}/{jid}_spec.npz"
    np.savez(out, freq=freq, freq_cm1=freq, sigma=sig, mean=mean, sigma_mean=smean, replicas=reps, prefactor_T=T_nve[reps],
             qm=QM, clip=CLIP, dipole=mu.astype(np.float32), T_nve=T_nve[reps], intact=intact, n_intact=len(reps),
             dt_fs=dt_fs, T=float(temp) if temp else np.nan, uid=uid, atomic_numbers=num,
             dipole_model=os.path.abspath(a.dipole), worker=a.who)
    R.put(out, f"dipoles/{jid}_spec.tmp.npz")
    tmp, fin = R.path(f"dipoles/{jid}_spec.tmp.npz"), R.path(f"dipoles/{jid}_spec.npz")
    v = R.sh(f"{a.remote_python} -c {shlex.quote(VERIFY)} {shlex.quote(tmp)} && mv {shlex.quote(tmp)} {shlex.quote(fin)}")
    if not a.keep:
        R.sh(f"rm -f {shlex.quote(R.path(f'traj/{jid}.npz'))}")
        if a.host != "local":
            os.remove(local)
    os.remove(out)
    wall = time.time() - t0
    rec = dict(id=jid, N=len(num), R=len(intact), n_intact=len(reps), frames=int(mu.shape[1]), pull_s=round(t_pull, 1),
               eval_s=round(t_eval, 1), wall_s=round(wall, 1), frames_per_s=round(mu.shape[0] * mu.shape[1] / max(t_eval, 1e-9)),
               verify=v.stdout.strip(), worker=a.who, end=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    R.sh(f"echo {shlex.quote(json.dumps(rec))} > {shlex.quote(R.path(f'claims/{jid}.lock/done.txt'))}", check=False)
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(f"[{rec['end']}] {jid}: N={rec['N']} intact {rec['n_intact']}/{rec['R']}, pull {t_pull:.0f} s, eval {t_eval:.0f} s "
          f"({rec['frames_per_s']} frames/s), total {wall:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dipole", required=True, help="frozen polar_v7 polar.model")
    ap.add_argument("--host", default="gh200", help="ssh host of the staging root, or 'local' to work on it in place")
    ap.add_argument("--root", default="/var/tmp/jjlee-pah/d1/cation_mlmd")
    ap.add_argument("--scratch", default=os.path.expanduser("~/pah/cation_dipole/scratch"))
    ap.add_argument("--remote-python", default="python3", help="python with numpy on the root host (spec check)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--max-jobs", type=int, default=0, help="stop after this many jobs (0 = until ALL_MD_DONE)")
    ap.add_argument("--only", default=None, help="claim only this id (smoke test)")
    ap.add_argument("--keep", action="store_true", help="do not delete traj/<id>.npz or the local copy")
    ap.add_argument("--log", default=None, help="jsonl of per-job timings (default <scratch>/../worker_<host>_gpu<k>.jsonl)")
    a = ap.parse_args()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    a.who = f"{socket.gethostname()}:gpu{gpu}:pid{os.getpid()}"
    a.scratch = os.path.join(a.scratch, f"gpu{gpu}")
    os.makedirs(a.scratch, exist_ok=True)
    logp = a.log or os.path.join(os.path.dirname(os.path.dirname(a.scratch)), f"worker_{socket.gethostname()}_gpu{gpu}.jsonl")
    log = open(logp, "a")
    R = Root(a.host, a.root); ev_cache = {}; done = 0
    print(f"{a.who}: root {a.host}:{a.root}, scratch {a.scratch}, B={a.batch}, model {a.dipole}", flush=True)
    while True:
        ready, claimed, all_done = R.listing()
        todo = sorted(ready - claimed)
        if a.only:
            todo = [j for j in todo if j == a.only]
        random.shuffle(todo)                                  # spread workers over the queue; mkdir decides
        got = None
        for jid in todo:
            if R.claim(jid, f"{a.who} {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"):
                got = jid; break
        if got:
            process(a, R, got, ev_cache, log); done += 1
            if (a.max_jobs and done >= a.max_jobs) or a.only:
                break
            continue
        if all_done and not (ready - claimed):
            break
        time.sleep(a.poll)
    ready, claimed, _ = R.listing()
    stale = R.sh(f"cd {shlex.quote(R.root)}/claims && for d in *.lock; do [ -e $d/done.txt ] || echo $d; done", check=False).stdout.split()
    print(f"{a.who}: exit after {done} jobs; traj left {len(ready)}; locks without done.txt (running or dead): {stale}", flush=True)


if __name__ == "__main__":
    main()
