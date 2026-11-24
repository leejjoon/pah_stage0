# IP-referenced cation energies for MLFF training (docs/plan/charged_mlff_plan.md, status log 2026-11-17): a sum of
# atomic terms cannot hold the ionisation energy, which is roughly size-independent (run1: the q1 energy error was a
# per-molecule offset tracking c*N - IP). This writes a copy of a dataset (d1/mlff_dataset.py) whose q1 frames carry
#   dft_energy = E_q1 - IP,  IP = E_q1(min) - E_q0(min) of the same pair (adiabatic, DFT),
# so the cation head learns a surface anchored at the neutral minimum; the raw value is kept as dft_energy_q1_raw and
# the subtracted IP as ip_ref (eV). q0 files are copied unchanged. Forces are unchanged: the shift is a per-molecule
# constant. A model trained on this predicts cation energies up to that constant: E_q1 = E_model + IP(pair).
#   python mlff_ip_ref.py <dataset dir> <out dir> <min-frame source extxyz>[,<more>]
# The min frames are read from the QA'd inputs (the dataset itself may lack a min frame that was dropped by a flag,
# e.g. C36H16_7143_q0_f000, whose RKS energy is then used: its BS state is 5.6 mHa lower, an IP shift of 0.15 eV that
# is again a per-molecule constant). The first min frame per (pair, charge) wins, as in mlff_dataset.py.
import glob, json, os, re, shutil, sys

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def frames(path):
    with open(path) as f:
        while (line := f.readline()).strip():
            n = int(line); head = f.readline().rstrip("\n"); rows = [f.readline() for _ in range(n)]
            yield {k: v.strip('"') for k, v in KV.findall(head)}, head, rows


def main():
    ds, out, srcs = sys.argv[1:4]
    emin = {}
    for p in srcs.split(","):
        for info, _, _ in frames(p):
            if info.get("config_type") == "min":
                emin.setdefault((info["source"].rsplit("_q", 1)[0], int(info["charge"])), float(info["dft_energy"]))
    os.makedirs(out, exist_ok=True); ips = {}
    for path in sorted(glob.glob(f"{ds}/*.extxyz")):
        dst = os.path.join(out, os.path.basename(path))
        if not path.endswith("_q1.extxyz"):
            shutil.copy(path, dst); continue
        with open(dst, "w") as f:
            for info, head, rows in frames(path):
                pair = info["source"].rsplit("_q", 1)[0]
                if (pair, 0) not in emin or (pair, 1) not in emin:
                    sys.exit(f"{path}: no min frame for {pair} in both charges")
                ip = emin[(pair, 1)] - emin[(pair, 0)]; ips[pair] = ip
                e = float(info["dft_energy"])
                head = re.sub(r" dft_energy=\S+", f" dft_energy={e - ip:.8f} dft_energy_q1_raw={e:.8f} ip_ref={ip:.6f}",
                              head, count=1)
                f.write(f"{len(rows)}\n{head}\n"); f.writelines(rows)
    json.dump({"ip_eV": ips, "source": srcs.split(",")}, open(os.path.join(out, "ip_ref.json"), "w"), indent=1)
    v = sorted(ips.values())
    print(f"{len(ips)} pairs; IP {v[0]:.2f}-{v[-1]:.2f} eV (median {v[len(v) // 2]:.2f}) -> {out}")


if __name__ == "__main__":
    main()
