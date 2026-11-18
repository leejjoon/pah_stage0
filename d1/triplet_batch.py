# Batch recomputation of PAHdb v4.00 non-Kekule species in their high-spin state (d1/triplet_freq.py --opt) on one GPU.
# Species and cost estimates: d1/triplet_species_v1.csv (sorted by estimate, largest first). Shard assignment is computed
# from that fixed list only (largest-first greedy balancing over n_shards), never from what is already done, so a resubmit
# with the same number of shards runs each species on the same shard and never twice at once.
#   python triplet_batch.py <species csv> <pahdb xml.gz> <out dir> --shard k --nshards N --hours H [--uids a,b,...]
# Finished species (<out>/<uid>.json) are skipped; a species starts only if 1.3 x its estimate fits in the time left;
# each runs as its own process (log <out>/logs/<uid>.log); a failure writes <out>/failed/<uid>.txt and the shard goes on.
import argparse, csv, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def assign(rows, n):
    load = [0.0] * n; shard = {}
    for r in rows:                                   # rows sorted largest first
        k = min(range(n), key=lambda i: load[i]); shard[r["uid"]] = k; load[k] += float(r["est_s"])
    return shard


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("species"); ap.add_argument("xml"); ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--hours", type=float, default=23.5); ap.add_argument("--uids", default=None, help="only these (testing)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.species)))
    if a.uids:
        keep = a.uids.split(","); rows = [r for r in rows if r["uid"] in keep]
    mine = [r for r in rows if assign(rows, a.nshards)[r["uid"]] == a.shard]
    os.makedirs(f"{a.out}/logs", exist_ok=True); os.makedirs(f"{a.out}/failed", exist_ok=True)
    deadline = time.time() + a.hours * 3600
    todo = [r for r in mine if not os.path.exists(f"{a.out}/{r['uid']}.json")]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} species, {len(todo)} to do, "
          f"estimate {sum(float(r['est_s']) for r in todo) / 3600:.1f} h, time {a.hours:.2f} h", flush=True)
    for r in todo:
        left = deadline - time.time()
        if 1.3 * float(r["est_s"]) > left:
            print(f"  stop: {r['uid']} {r['formula']} needs ~{float(r['est_s']) / 3600:.1f} h, {left / 3600:.1f} h left", flush=True)
            break
        t = time.time()
        cmd = [sys.executable, "-I", f"{HERE}/triplet_freq.py", a.xml, r["uid"], f"{a.out}/{r['uid']}.json", "--opt",
               "--mult", r["mult"]]
        with open(f"{a.out}/logs/{r['uid']}.log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        dt = (time.time() - t) / 3600
        if rc == 0 and os.path.exists(f"{a.out}/{r['uid']}.json"):
            print(f"  done {r['uid']} {r['formula']} in {dt:.2f} h (estimate {float(r['est_s']) / 3600:.2f})", flush=True)
        else:
            open(f"{a.out}/failed/{r['uid']}.txt", "w").write(f"exit {rc} after {dt:.2f} h; see logs/{r['uid']}.log\n")
            print(f"  FAILED {r['uid']} {r['formula']} (exit {rc}) after {dt:.2f} h", flush=True)
    left = [r for r in mine if not os.path.exists(f"{a.out}/{r['uid']}.json")]
    print(f"shard {a.shard}: {len(mine) - len(left)}/{len(mine)} done", flush=True)


if __name__ == "__main__":
    main()
