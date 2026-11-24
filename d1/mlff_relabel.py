# Swap new DFT labels into QA'd extxyz files (dataset v2, docs/plan/charged_mlff_plan.md): frames are matched by
# `log` name; the new file must hold the same species and positions (checked to 1e-6 A). Usage:
#   python mlff_relabel.py <qa.extxyz> <new labels.extxyz>[,<more>] <out.extxyz> [--missed-instability NAME,...]
# For a matched frame the label keys below and the force columns come from the new file, plus its bs_label
# (d1/dft_seed_run.py collect --bs-label); everything else (source, config_type, xtb_*, ...) stays. Unmatched frames
# are copied unchanged. --missed-instability names frames whose stability test passed although an instability is
# known (e.g. C36H16_7143_q0_f000, found by the 2026-11-15 minima check): their stability is set to
# unstable_rhf_uhf, so mlff_qa.py flags them `unstable`. Re-run `mlff_qa.py --reuse-xtb` on the output afterwards
# so that qa_flags follow the new labels.
import argparse, re, sys
import numpy as np

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
LABEL_KEYS = ("dft_energy", "dft_dipole", "s2", "stability", "scf_cycles", "method", "converged", "newton", "bs_label")


def frames(path):
    with open(path) as f:
        while (line := f.readline()).strip():
            n = int(line); head = f.readline().rstrip("\n"); rows = [f.readline() for _ in range(n)]
            yield dict(KV.findall(head)), head, rows


def set_key(head, k, v):
    if re.search(rf" {k}=", head):
        return re.sub(rf' {k}=("[^"]*"|\S+)', lambda _: f" {k}={v}", head)
    return head.replace(" log=", f" {k}={v} log=", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qa"); ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--missed-instability", default="")
    a = ap.parse_args()
    missed = {x for x in a.missed_instability.split(",") if x}
    new = {}
    for p in a.new.split(","):
        for info, head, rows in frames(p):
            new[info["log"]] = (info, rows)
    n_in = n_rel = 0; used = set()
    with open(a.out, "w") as out:
        for info, head, rows in frames(a.qa):
            n_in += 1; name = info["log"]
            if name in new:
                ni, nrows = new[name]
                old = np.array([[float(x) for x in r.split()[1:4]] for r in rows])
                nw = np.array([[float(x) for x in r.split()[1:4]] for r in nrows])
                if [r.split()[0] for r in rows] != [r.split()[0] for r in nrows] or np.abs(old - nw).max() > 1e-6:
                    sys.exit(f"{name}: species or positions differ between {a.qa} and the new labels")
                for k in LABEL_KEYS:
                    head = set_key(head, k, ni[k])
                rows = nrows; used.add(name); n_rel += 1
            if name in missed:
                head = set_key(head, "stability", "unstable_rhf_uhf"); head = set_key(head, "bs_label", "missed_instability")
                used.add(name)
            out.write(f"{len(rows)}\n{head}\n"); out.writelines(rows)
    print(f"{a.qa}: {n_in} frames, {n_rel} relabelled -> {a.out}")
    return used


if __name__ == "__main__":
    main()
