#!/usr/bin/env python3
"""T0 timing grid (plan 4.4): runs every cell through the drivers inside the images, on GPU 0 pinned to NUMA node 0,
and writes one CSV row per run with the plan's columns plus what the host sees.

Runs on the host (stdlib only). For each cell it
  - waits until no other user's process is on GPU 0 (plan: "nothing else on the node"), unless --allow-shared;
    the row records how many foreign GPU processes were present at start and end (node_foreign_procs)
  - samples `nvidia-smi dmon` (GPU util %, memory) and /proc/stat for CPUs 0-71 during the run
  - lets the driver append its row to a per-run CSV, then adds gpu_util (mean SM % over the run), gpu_mem_gb_smi
    (max memory.used, whole device), cpu_util (mean % of the 72 pinned cores), wall_total_s, node state, image hashes
  - skips cells already in the output CSV (resumable)

gpu_util and cpu_util are averaged over the whole process lifetime (model load, compile and equilibration included),
so they describe the run, not just the timed window. wall_s/s_per_mol_ps come from the driver (production only).

Usage:
  t0/grid.py --phase calib                      # short runs (100 + 400 steps), 1 repeat, shared GPU allowed
  t0/grid.py --phase full [--repeats 3]         # the benchmark: 2,000 + 20,000 steps
  t0/grid.py --phase full --only 'mode=torchsim,batch=100' --dry-run
"""
import argparse, csv, hashlib, itertools, json, os, pwd, re, shlex, subprocess, sys, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "/var/tmp/jjlee-pah"
SIF = {"jax": f"{ROOT}/sif/A_jax.sif", "torch": f"{ROOT}/sif/B_torchsim.sif"}
BUNDLES = f"{ROOT}/eq64"
JAX_BUNDLE = {"off": "off_small_f32", "cueq": "off_small_f32_cueq", "cueqf": "off_small_f32_cueqf"}
PIN = ["numactl", "--cpunodebind=0", "--membind=0"]
CPUS = range(0, 72)
COLS = ["cell", "stack", "mode", "batched_impl", "cueq", "cueq_version", "conv_fusion", "dtype", "matmul_precision", "batch",
        "mps_workers", "repeat", "species_set", "ensemble", "steps", "wall_s", "s_per_mol_ps", "ns_per_day_per_traj", "gpu_util",
        "gpu_mem_gb", "gpu_mem_gb_smi", "cpu_util", "compile_s", "edge_cap", "cuex", "wall_total_s", "node_foreign_procs",
        "phase", "started", "host", "driver", "sif_sha256", "log"]

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=["calib", "full"])
ap.add_argument("--repeats", type=int, default=None, help="default: 1 for calib, 3 for full")
ap.add_argument("--workers", default="10:4;100:16,32", help="MPS pool sizes P per batch, e.g. 10:4;100:16,32 "
                "(P=4 at batch 100 was dropped on 2026-11-09: 10.4 s/mol-ps vs 3.1-3.7 at P=16/32 in calibration, 7.5 h per repeat)")
ap.add_argument("--only", default="", help="comma list of key=value filters on cell fields, e.g. mode=mps,batch=100")
ap.add_argument("--out", default=None, help="output CSV (default reports/t0_timing_<phase>.csv)")
ap.add_argument("--md", default=f"{ROOT}/md/grid", help="scratch dir for npz files and logs")
ap.add_argument("--allow-shared", action="store_true", help="do not wait for GPU 0 to be free of other users")
ap.add_argument("--wait-max", type=float, default=24 * 3600, help="max seconds to wait for a free GPU per cell")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
reps = args.repeats or (1 if args.phase == "calib" else 3)
EQUIL, PROD = (100, 400) if args.phase == "calib" else (2000, 20000)
out = args.out or os.path.join(REPO, "reports", f"t0_timing_{args.phase}.csv")
P_BY_B = {int(k): [int(p) for p in v.split(",")] for k, v in (kv.split(":") for kv in args.workers.split(";"))}


def cells():
    """Every timed configuration. Sequential is measured at 1 and 10 (plan: extrapolated to 100); the MPS pool at 10 and
    100 (at 1 it is sequential); JAX runs all three bundles so the committee's cueq-on choice needs no rerun."""
    for r in range(reps):
        for b in (1, 10, 100):
            for bund in ("off", "cueq", "cueqf"):
                yield dict(stack="jax", mode="batched", impl="jax", cueq=bund, batch=b, P="", repeat=r)
            for cq in ("no", "yes"):
                if b <= 10: yield dict(stack="torch", mode="sequential", impl="", cueq=cq, batch=b, P="", repeat=r)
                for impl in ("torchsim", "concat"):
                    yield dict(stack="torch", mode="batched", impl=impl, cueq=cq, batch=b, P="", repeat=r)
                if b >= 10:
                    for P in P_BY_B.get(b, []):
                        if P <= b: yield dict(stack="torch", mode="mps_pool", impl="", cueq=cq, batch=b, P=P, repeat=r)


def name(c):
    return f"{c['stack']}-{c['mode']}{'-' + c['impl'] if c['impl'] and c['impl'] != 'jax' else ''}-cueq_{c['cueq']}-b{c['batch']}" \
           f"{'-P' + str(c['P']) if c['P'] else ''}-r{c['repeat']}"


def command(c, run_csv, prefix):
    common = ["--batch", str(c["batch"]), "--equil", str(EQUIL), "--prod", str(PROD), "--repeat", str(c["repeat"]),
              "--csv", "/md/" + os.path.basename(run_csv), "--out", "/md/" + os.path.basename(prefix)]
    if c["stack"] == "jax":
        env = ["--env", "JAX_PLATFORMS=cuda,JAX_DEFAULT_MATMUL_PRECISION=highest,XLA_PYTHON_CLIENT_PREALLOCATE=false"]
        return PIN + ["apptainer", "exec", "--nv", *env, "--bind", f"{REPO}:/r", "--bind", f"{args.md}:/md",
                      "--bind", f"{BUNDLES}:/b", SIF["jax"], "python", "/r/t0/md_jax.py",
                      "--bundle", f"/b/{JAX_BUNDLE[c['cueq']]}.npz", *common]
    extra = ["--cueq"] if c["cueq"] == "yes" else []
    if c["mode"] == "mps_pool":
        return [os.path.join(REPO, "t0/run_mps.sh"), SIF["torch"], *common, *extra, "--workers", str(c["P"])]
    mode = c["impl"] if c["mode"] == "batched" else "sequential"
    return PIN + ["apptainer", "exec", "--nv", "--bind", f"{REPO}:/r", "--bind", f"{args.md}:/md", SIF["torch"],
                  "python", "/r/t0/md_torch.py", "--mode", mode, *common, *extra]


def foreign_gpu0():
    """Processes of other users on GPU 0 (nvidia-smi index 0)."""
    q = subprocess.run(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True).stdout
    me, n = os.getuid(), 0
    for pid in q.split():
        try:
            if os.stat(f"/proc/{pid}").st_uid != me: n += 1
        except FileNotFoundError: n += 1          # another PID namespace or already gone: count it, be conservative
    return n


def cpu_times():
    t = {}
    for line in open("/proc/stat"):
        m = re.match(r"cpu(\d+) (.*)", line)
        if m and int(m.group(1)) in CPUS:
            v = list(map(int, m.group(2).split())); t[int(m.group(1))] = (sum(v), v[3] + v[4])   # total, idle+iowait
    return t


class Dmon(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.sm, self.mem = [], []
        self.p = subprocess.Popen(["nvidia-smi", "dmon", "-i", "0", "-s", "um", "-d", "1"], stdout=subprocess.PIPE, text=True)
    def run(self):
        hdr = None
        for line in self.p.stdout:
            if line.startswith("#"):
                if hdr is None: hdr = line[1:].split()
                continue
            v = line.split()
            if hdr and len(v) == len(hdr):
                d = dict(zip(hdr, v))
                try: self.sm.append(float(d["sm"])); self.mem.append(float(d.get("fb", "nan")))
                except ValueError: pass
    def stop(self):
        self.p.terminate()


def sha(path, cache={}):
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
        cache[path] = h.hexdigest()[:16]
    return cache[path]


def done_cells():
    if not os.path.exists(out): return set()
    return {r["cell"] for r in csv.DictReader(open(out))}


def main():
    os.makedirs(args.md, exist_ok=True); os.makedirs(os.path.dirname(out), exist_ok=True)
    flt = dict(kv.split("=", 1) for kv in args.only.split(",") if kv)
    todo = [c for c in cells() if all(str(c.get({"impl": "impl"}.get(k, k), "")) == v for k, v in flt.items())]
    done = done_cells()
    todo = [c for c in todo if name(c) not in done]
    print(f"{len(todo)} cells to run ({len(done)} already in {out}); equil {EQUIL}, prod {PROD}", flush=True)
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"], capture_output=True, text=True).stdout.strip()
    host = os.uname().nodename
    for i, c in enumerate(todo):
        n = name(c); prefix = os.path.join(args.md, f"{args.phase}-{n}"); run_csv = prefix + ".row.csv"; log = prefix + ".log"
        cmd = command(c, run_csv, prefix)
        if args.dry_run: print(n, "\n   ", shlex.join(cmd)); continue
        t_wait = time.time()
        while not args.allow_shared and foreign_gpu0():
            if time.time() - t_wait > args.wait_max: sys.exit(f"GPU 0 still busy after {args.wait_max:.0f} s; stopping before {n}")
            time.sleep(60)
        if os.path.exists(run_csv): os.remove(run_csv)
        f0 = foreign_gpu0(); started = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"[{i + 1}/{len(todo)}] {started} {n} (foreign GPU0 procs {f0})", flush=True)
        dm = Dmon(); dm.start(); c0 = cpu_times(); t0 = time.time()
        with open(log, "w") as lf:
            lf.write(shlex.join(cmd) + "\n"); lf.flush()
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "MD_DIR": args.md}).returncode
        t1 = time.time(); c1 = cpu_times(); dm.stop(); f1 = foreign_gpu0()
        if rc != 0 or not os.path.exists(run_csv):
            print(f"   FAILED rc={rc}; see {log}", flush=True); continue
        row = list(csv.DictReader(open(run_csv)))[-1]
        busy = [1 - (c1[k][1] - c0[k][1]) / max(1, c1[k][0] - c0[k][0]) for k in c0]
        row.update(cell=n, gpu_util=round(sum(dm.sm) / max(1, len(dm.sm)), 1),
                   gpu_mem_gb_smi=round(max(dm.mem, default=0) / 1024, 3), cpu_util=round(100 * sum(busy) / len(busy), 1),
                   wall_total_s=round(t1 - t0, 1), node_foreign_procs=f"{f0}/{f1}", phase=args.phase, started=started,
                   host=host, driver=driver, sif_sha256=sha(SIF[c["stack"]]), log=os.path.basename(log))
        if c["stack"] == "jax": row["cueq"] = {"off": "no", "cueq": "backend_only", "cueqf": "conv_fusion"}[c["cueq"]]
        new = not os.path.exists(out)
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); new and w.writeheader(); w.writerow(row)
        print(f"   {row.get('s_per_mol_ps')} s/mol-ps, wall {row.get('wall_s')} s (total {t1 - t0:.0f} s), "
              f"GPU {row['gpu_util']}%, CPU {row['cpu_util']}%", flush=True)


if __name__ == "__main__":
    main()
