# Curated training data for the charged-state MLFFs (docs/plan/charged_mlff_plan.md, Phase 2), framework-neutral:
# one extended XYZ per split and charge (mace-jax reads it directly, keys dft_energy / dft_forces / dft_dipole; a
# NeuralIL/EPNN loader reads the same files). Frames are copied verbatim (every label and qa_flags kept).
#   python mlff_dataset.py <qa.extxyz>[,<more>] <species.csv>[,<more>] <out dir>
#        [--exclude d1/seed_exclusions_open_shell.csv | /dev/null] [--drop-flags unstable,...] [--drop-frames <csv>,...] [--keep-frames <csv>,...]
# Steps, with frame counts per split and charge after each (report.md / report.json in <out dir>):
#   1. exclusions: pairs from the exclusion list, in every split (Phase 1 decision 2026-11-15). An optional `charges`
#      column ("0", "1" or "0;1") limits an entry to those charges; without it (or empty) the whole pair goes. Several
#      lists can be given, comma-separated;
#   2. duplicates: a later dataset's min frame for a (pair, charge) already present is dropped, after checking that
#      the centred positions agree to 1e-6 A (otherwise the run stops);
#   3. --drop-flags: frames carrying any of these qa_flags (default: none; the choice is the user's and the report
#      states it). One flag is computed here from the geometry rather than read from qa_flags: `short_ch`, a C-H
#      distance below SHORT_CH (0.75 A). mlff_sample.py's bound_oop push follows the molecular-plane normal, which in
#      sp3 / non-planar molecules can drive an H into its carbon (0.62-0.74 A seen; 2026-11-21, dropped by decision).
#      --drop-frames: single frames dropped by decision, CSV with columns frame (the frame's `log` key, e.g.
#      C10H10_x02_q1_f007), reason; counted as flag `dropped_frame`. Every listed frame must be found (else the run stops).
#      --keep-frames: exceptions to --drop-flags by decision, CSV with columns frame, flag, reason: that frame is not
#      dropped for that flag (other flags still apply). Every listed frame must be found.
# The input order of the extxyz files decides which copy of a duplicate is kept (the first).
import argparse, collections, csv, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlff_coverage import SOLO_FRAC  # noqa: E402

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
SPLITS = ["train", "validation", "final_test", "transfer_test"]
SHORT_CH = 0.75   # A


def read_frames(path):
    with open(path) as f:
        while True:
            line = f.readline()
            if not line.strip():
                return
            n = int(line); comment = f.readline(); atoms = [f.readline() for _ in range(n)]
            info = {k: v.strip('"') for k, v in KV.findall(comment)}
            yield (info, line + comment + "".join(atoms), np.array([[float(x) for x in a.split()[1:4]] for a in atoms]),
                   np.array([a.split()[0] for a in atoms]))


def min_ch(sym, pos):
    c, h = pos[sym == "C"], pos[sym == "H"]
    return float(np.linalg.norm(c[:, None] - h[None], axis=-1).min()) if len(c) and len(h) else np.inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extxyz"); ap.add_argument("species"); ap.add_argument("out")
    ap.add_argument("--exclude", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "seed_exclusions_open_shell.csv"))
    ap.add_argument("--drop-flags", default="")
    ap.add_argument("--drop-frames", default="")
    ap.add_argument("--keep-frames", default="")
    a = ap.parse_args()
    drop = {x for x in a.drop_flags.split(",") if x}
    drop_frames = {r["frame"] for p in a.drop_frames.split(",") if p for r in csv.DictReader(open(p))}
    if drop_frames:
        drop.add("dropped_frame")
    found = set()
    keep = collections.defaultdict(set)
    for p in (x for x in a.keep_frames.split(",") if x):
        for r in csv.DictReader(open(p)):
            keep[r["frame"]].add(r["flag"])
    kept_found = set()
    sp = {}
    for p in a.species.split(","):
        sp.update({r["pair"]: r for r in csv.DictReader(open(p))})
    excl_q = collections.defaultdict(set)                    # pair -> excluded charges
    for path in a.exclude.split(","):
        for r in csv.DictReader(open(path)):
            qs = (r.get("charges") or "").strip()
            excl_q[r["pair"]] |= {int(x) for x in qs.split(";")} if qs else {0, 1}
    excl = {p for p, qs in excl_q.items() if qs >= {0, 1}}  # whole pairs (for the solo/H bias line)
    stage = collections.OrderedDict((s, collections.Counter()) for s in ("input", "after_exclusions", "after_dedup",
                                                                         "after_flags"))
    by_ct = collections.Counter(); dropped_flags = collections.Counter(); seen_min = {}; kept = collections.defaultdict(list)
    for path in a.extxyz.split(","):
        for info, text, pos, sym in read_frames(path):
            pair = info["source"].rsplit("_q", 1)[0]; q = int(info["charge"])
            if pair not in sp:
                sys.exit(f"{path}: pair {pair} not in the species tables")
            key = (sp[pair]["split"], q); stage["input"][key] += 1
            if q in excl_q.get(pair, ()):
                continue
            stage["after_exclusions"][key] += 1
            if info.get("config_type") == "min":
                if (pair, q) in seen_min:
                    if np.abs(seen_min[(pair, q)] - (pos - pos.mean(0))).max() > 1e-6:
                        sys.exit(f"{path}: min frame of {pair} q{q} differs from the first copy")
                    continue
                seen_min[(pair, q)] = pos - pos.mean(0)          # centred: samplers may centre differently (~1e-5 A)
            stage["after_dedup"][key] += 1
            flags = set(info.get("qa_flags", "ok").split(","))
            if info.get("log") in drop_frames:
                flags.add("dropped_frame"); found.add(info["log"])
            if "short_ch" in drop and min_ch(sym, pos) < SHORT_CH:
                flags.add("short_ch")
            if info.get("log") in keep:
                kept_found.add(info["log"]); flags -= keep[info["log"]]
            if flags & drop:
                for fl in flags & drop:
                    dropped_flags[(key, fl)] += 1
                continue
            stage["after_flags"][key] += 1; by_ct[key + (info.get("config_type", "?"),)] += 1
            kept[key].append(text)
    if set(keep) - kept_found:
        sys.exit(f"--keep-frames: not found: {sorted(set(keep) - kept_found)}")
    if drop_frames - found:
        sys.exit(f"--drop-frames: not found: {sorted(drop_frames - found)}")
    os.makedirs(a.out, exist_ok=True)
    for (s, q), frames in sorted(kept.items()):
        with open(f"{a.out}/{s}_q{q}.extxyz", "w") as f:
            f.writelines(frames)
    keys = [(s, q) for s in SPLITS for q in (0, 1)]
    med = lambda ps: sorted(SOLO_FRAC(sp[p]) for p in ps)[len(ps) // 2] if ps else float("nan")
    tr = [p for p, r in sp.items() if r["split"] == "train"]
    bias = (f"median solo/H fraction of the train pairs: kept {med([p for p in tr if p not in excl]):.2f}, excluded "
            f"{med([p for p in tr if p in excl]):.2f}")
    lines = ["# Curated MLFF training data", "", f"Inputs: {a.extxyz}", f"Species: {a.species}",
             f"Exclusions: {a.exclude} ({len(excl)} pairs)", f"Dropped qa_flags: {sorted(drop) or 'none'}", "",
             ("**Selection bias (state wherever these data are reported):** the open-shell exclusion removes "
              f"solo-H-rich pairs: {bias}. The cut acts as a solo-H and size selection (the Wang 2026 Fig. 1 bias)."
              if excl else "No pairs excluded (no open-shell selection bias from exclusions)."), "",
             "## Frames per split and charge after each step", "",
             "| step | " + " | ".join(f"{s} q{q}" for s, q in keys) + " |", "|---|" + "---|" * len(keys)]
    lines += [f"| {st} | " + " | ".join(str(c[k]) for k in keys) + " |" for st, c in stage.items()]
    if dropped_flags:
        lines += ["", "## Frames dropped per flag (a frame with two dropped flags counts twice)", "",
                  "| flag | " + " | ".join(f"{s} q{q}" for s, q in keys) + " |", "|---|" + "---|" * len(keys)]
        for fl in sorted({f for _, f in dropped_flags}):
            lines.append(f"| {fl} | " + " | ".join(str(dropped_flags[(k, fl)]) for k in keys) + " |")
    cts = sorted({k[2] for k in by_ct})
    lines += ["", "## Kept frames per config_type", "", "| config_type | " + " | ".join(f"{s} q{q}" for s, q in keys)
              + " |", "|---|" + "---|" * len(keys)]
    lines += [f"| {c} | " + " | ".join(str(by_ct[k + (c,)]) for k in keys) + " |" for c in cts]
    open(f"{a.out}/report.md", "w").write("\n".join(lines) + "\n")
    json.dump(dict(inputs=a.extxyz.split(","), species=a.species.split(","), exclude=a.exclude, drop_flags=sorted(drop),
                   bias=bias, stages={st: {f"{s}_q{q}": c[(s, q)] for s, q in keys} for st, c in stage.items()},
                   files={f"{s}_q{q}": len(kept[(s, q)]) for s, q in keys}),
              open(f"{a.out}/report.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
