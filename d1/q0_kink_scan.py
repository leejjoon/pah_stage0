# q0 Coulson–Fischer kink scan (course correction 2026-11-23, item 2): relaxed C–C stretch scans of the neutral (q0
# head) for every C–C bond with an sp3 carbon, per model; container B, CPU is enough:
#   python q0_kink_scan.py <out.json> --xyz a_q0.xyz,b_q0.xyz --models v7=<model>,v8=<model> [--r 1.40:3.00:0.05] [--device cpu]
#                          [--bonds C4-C8] [--save-geom scan.extxyz]
# Per bond: from the minimum, the bond is set to each r (ASE FixBondLength, the other atoms relaxed with BFGS to
# fmax 0.02 eV/A, each point started from the previous one), energy relative to the minimum. A spin-blind q0 head
# trained on RKS labels near the minimum and BS-UKS labels once the bond is stretched can show a kink where the
# labels switch (the RKS->BS Coulson–Fischer point, ~1.9–2.2 A for a C–C bond). Reported per bond and model: the
# largest |second difference| of E(r) beyond 1.6 A, in eV/A^2, against the median there (a kink stands out as a
# spike), the r where it is, and the energy at 3.0 A.
import argparse, json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_integrity import bonds, read_xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--xyz", required=True); ap.add_argument("--models", required=True)
    ap.add_argument("--r", default="1.40:3.00:0.05"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--bonds", default="", help="only these bonds, e.g. C4-C8 (all sp3 C-C bonds by default)")
    ap.add_argument("--save-geom", default="", help="write each scan point of the first model as extxyz frames (mlff_sample.py format, for DFT)")
    a = ap.parse_args()
    from ase import Atoms
    from ase.constraints import FixBondLength
    from ase.optimize import BFGS
    from mace.calculators import MACECalculator
    r0, r1, dr = map(float, a.r.split(":")); rs = np.arange(r0, r1 + dr / 2, dr)
    models = dict(x.split("=", 1) for x in a.models.split(","))
    calcs = {m: MACECalculator(model_paths=p, device=a.device, default_dtype="float64", head="q0") for m, p in models.items()}
    out = {}
    for f in a.xyz.split(","):
        sym, pos = read_xyz(f); name = os.path.basename(f)[:-4]
        bl = bonds(sym, pos); deg = {i: sum(i in b for b in bl) for i in range(len(sym))}
        cc = sorted(tuple(sorted(b)) for b in bl if sym[b[0]] == "C" and sym[b[1]] == "C"
                    and (deg[b[0]] == 4 or deg[b[1]] == 4))
        if a.bonds:
            cc = [b for b in cc if f"C{b[0]}-C{b[1]}" in a.bonds.split(",")]
        for i, j in cc:
            for m, calc in calcs.items():
                at = Atoms(sym, positions=pos); at.calc = calc
                BFGS(at, logfile=None).run(fmax=0.01, steps=500); e_min = at.get_potential_energy(); p0 = at.positions.copy()
                E = []
                for r in rs:
                    v = at.positions[j] - at.positions[i]; d = np.linalg.norm(v)
                    at.set_constraint(); at.positions[j] += v / d * (r - d)      # move j along the bond to the new length
                    at.set_constraint(FixBondLength(i, j)); BFGS(at, logfile=None).run(fmax=0.02, steps=300)
                    E.append(at.get_potential_energy() - e_min)
                    if a.save_geom and m == list(calcs)[0]:
                        with open(a.save_geom, "a") as g:
                            g.write(f"{len(sym)}\nProperties=species:S:1:pos:R:3 charge=0 mult=1 source={name}.xyz config_type=scan_C{i}C{j} "
                                    f"sample_T=none max_stretch={r:.3f} max_elong=0.0000 graph_changed={int(r > 1.2 * 1.52)}\n")
                            g.write("".join(f"{e} {x:.8f} {y:.8f} {z:.8f}\n" for e, (x, y, z) in zip(sym, at.positions)))
                E = np.array(E); d2 = np.diff(E, 2) / dr ** 2; k = rs[1:-1] >= 1.6
                kk = np.argmax(np.abs(d2) * k)
                rec = dict(r=rs.tolist(), E=E.tolist(), max_abs_d2=float(abs(d2[kk])), r_at_max=float(rs[1:-1][kk]),
                           median_abs_d2=float(np.median(np.abs(d2[k]))), E_at_end=float(E[-1]))
                out[f"{name} C{i}-C{j} {m}"] = rec
                print(f"{name} C{i}-C{j} {m}: E(3.0) {E[-1]:.2f} eV, max |d2E| {rec['max_abs_d2']:.1f} at {rec['r_at_max']:.2f} A "
                      f"(median {rec['median_abs_d2']:.1f}) eV/A^2", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
