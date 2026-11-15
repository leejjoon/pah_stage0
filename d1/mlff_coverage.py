# Coverage report for the charged-MLFF DFT data (docs/plan/charged_mlff_plan.md, "Quality checks", item 5), after the
# Phase 1 exclusions (d1/seed_exclusions_open_shell.csv). Reads the labelled extxyz (mlff_qa.py output preferred, for
# qa_flags; dft_seed_run.py collect output also works), the species splits and the exclusion list; prints Markdown
# tables and writes <out>_frames.csv (one row per split, charge, 10-C bin, config_type: frames kept / excluded).
#   python mlff_coverage.py <seed.extxyz> <species_v2.csv> <out prefix> [--exclude d1/seed_exclusions_open_shell.csv]
# Checks (exit status 1 if one fails): every frame's pair is in the species table; no carbon skeleton spans two
# splits; every excluded pair is in the species table; every pair has frames for both charges.
import argparse, collections, csv, os, re, sys

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
SOLO_FRAC = lambda s: int(s["n_solo"]) / max(1, sum(int(s[k]) for k in ("n_solo", "n_duo", "n_trio", "n_quartet")))


def frames(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line.strip():
                return
            n = int(line); info = {k: v.strip('"') for k, v in KV.findall(f.readline())}
            for _ in range(n):
                f.readline()
            yield info


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extxyz"); ap.add_argument("species"); ap.add_argument("out")
    ap.add_argument("--exclude", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "seed_exclusions_open_shell.csv"))
    a = ap.parse_args()
    sp = {r["pair"]: r for r in csv.DictReader(open(a.species))}
    excl = {r["pair"] for r in csv.DictReader(open(a.exclude))}
    fails = []
    skel = collections.defaultdict(set)
    for r in sp.values():
        skel[r["skeleton"]].add(r["split"])
    fails += [f"skeleton {k} in splits {sorted(v)}" for k, v in skel.items() if len(v) > 1]
    fails += [f"excluded pair {p} not in {a.species}" for p in excl - set(sp)]
    cnt = collections.Counter(); charges = collections.defaultdict(set); flags = collections.Counter()
    ctypes = set(); unknown = set()
    for info in frames(a.extxyz):
        pair = info["source"].rsplit("_q", 1)[0]
        if pair not in sp:
            unknown.add(pair); continue
        s = sp[pair]; q = int(info["charge"]); b = (int(s["nC"]) - 1) // 10 * 10 + 1
        key = (s["split"], q, f"{b}-{b + 9}", info.get("config_type", "?"), pair in excl)
        cnt[key] += 1; charges[pair].add(q); ctypes.add(key[3])
        for fl in filter(None, info.get("qa_flags", "").split(",")):
            if fl not in ("none", "ok"):
                flags[(s["split"], q, fl, pair in excl)] += 1
    fails += [f"pair {p} in the data but not in {a.species}" for p in sorted(unknown)]
    fails += [f"pair {p}: charges {sorted(c)}" for p, c in charges.items() if c != {0, 1}]
    fails += [f"pair {p} in {a.species} has no frames" for p in sorted(set(sp) - set(charges))]
    splits = ["train", "validation", "final_test", "transfer_test"]
    splits += sorted({k[0] for k in cnt} - set(splits))

    print("## Pairs and frames per split (kept / excluded)\n")
    rows = []
    for s in splits:
        pk = [p for p, r in sp.items() if r["split"] == s and p not in excl]; px = [p for p, r in sp.items()
                                                                                    if r["split"] == s and p in excl]
        f = lambda q, x: sum(v for k, v in cnt.items() if k[0] == s and k[1] == q and k[4] == x)
        med = lambda ps: sorted(SOLO_FRAC(sp[p]) for p in ps)[len(ps) // 2] if ps else float("nan")
        rows.append([s, len(pk), len(px), f(0, False), f(1, False), f(0, True) + f(1, True),
                     f"{med(pk):.2f}", f"{med(px):.2f}"])
    print(table(["split", "pairs kept", "pairs excl.", "frames q0", "frames q1", "frames excl.",
                 "median solo/H kept", "median solo/H excl."], rows))

    bins = sorted({k[2] for k in cnt}, key=lambda b: int(b.split("-")[0]))
    print("\n## Kept frames per 10-C bin (q0 / q1)\n")
    rows = []
    for b in bins:
        r = [b]
        for s in splits:
            g = lambda q: sum(v for k, v in cnt.items() if k[0] == s and k[1] == q and k[2] == b and not k[4])
            r.append(f"{g(0)} / {g(1)}" if g(0) + g(1) else "–")
        rows.append(r)
    print(table(["nC"] + splits, rows))

    order = sorted(ctypes, key=lambda c: (re.sub(r"\d", "", c), int(re.sub(r"\D", "", c) or 0)))
    print("\n## Kept frames per config_type (q0 / q1)\n")
    rows = []
    for c in order:
        r = [c]
        for s in splits:
            g = lambda q: sum(v for k, v in cnt.items() if k[0] == s and k[1] == q and k[3] == c and not k[4])
            r.append(f"{g(0)} / {g(1)}" if g(0) + g(1) else "–")
        rows.append(r)
    print(table(["config_type"] + splits, rows))

    if flags:
        print("\n## qa_flags on kept frames (q0 / q1); excluded pairs in brackets\n")
        rows = []
        for fl in sorted({k[2] for k in flags}):
            r = [fl]
            for s in splits:
                g = lambda q, x: flags[(s, q, fl, x)]
                r.append(f"{g(0, False)} / {g(1, False)}" + (f" [{g(0, True)} / {g(1, True)}]"
                                                              if g(0, True) + g(1, True) else ""))
            rows.append(r)
        print(table(["flag"] + splits, rows))

    with open(a.out + "_frames.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "charge", "nC_bin", "config_type", "excluded", "frames"])
        for k in sorted(cnt):
            w.writerow(list(k[:4]) + [int(k[4]), cnt[k]])
    print("\n## Checks\n")
    print("\n".join(f"- FAIL: {x}" for x in fails) if fails else
          "- all pairs known, both charges present, no skeleton spans two splits, exclusions valid: PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
