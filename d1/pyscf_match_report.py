# Summarise pyscf_match.py results: for each tag and setting, energy relative to a reference energy (PAHdb or a
# Gaussian log) and to the grid-converged PySCF setting (ref_200x974), the largest force-component and dipole
# differences from that setting, <S^2>, SCF cycles and time.
#   python pyscf_match_report.py <outdir> --ref naph_q0=-385.89270636 --ref naph_q1=-385.61361154 [--md]
import argparse, glob, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir"); ap.add_argument("--ref", action="append", default=[], help="tag=energy_Ha")
    ap.add_argument("--conv", default="ref_200x974"); ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    refs = {k: float(v) for k, v in (r.split("=") for r in a.ref)}
    R = {}
    for f in sorted(glob.glob(f"{a.outdir}/*.json")):
        r = json.load(open(f)); R.setdefault(r["tag"], {})[r["setting"]] = r
    hdr = ["setting", "dE_ref (uHa)", "dE_conv (uHa)", "max|dF| (Ha/bohr)", "|dmu| (D)", "<S2>", "cycles",
           "ngrids", "t (s)"]
    for tag, S in R.items():
        c = S.get(a.conv)
        print(f"\n{tag}: reference {refs.get(tag)}; converged-grid setting {a.conv if c else 'missing'}")
        rows = []
        for name, r in S.items():
            dref = (r["energy"] - refs[tag]) * 1e6 if tag in refs else np.nan
            if c:
                dc = (r["energy"] - c["energy"]) * 1e6
                df = np.abs(np.array(r["grad"]) - np.array(c["grad"])).max()
                dm = np.linalg.norm(np.array(r["dipole_debye"]) - np.array(c["dipole_debye"]))
            else:
                dc = df = dm = np.nan
            rows.append([name, f"{dref:+.1f}", f"{dc:+.1f}", f"{df:.1e}", f"{dm:.1e}", f"{r['s2']:.4f}",
                         str(r["scf_cycles"]) + ("+N" if r["newton"] else ""), str(r["ngrids"]), f"{r['t_total']:.1f}"])
        if a.md:
            print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
            for x in rows:
                print("| " + " | ".join(x) + " |")
        else:
            w = [max(len(h), *(len(x[i]) for x in rows)) for i, h in enumerate(hdr)]
            print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
            for x in rows:
                print("  ".join(v.ljust(w[i]) for i, v in enumerate(x)))


if __name__ == "__main__":
    main()
