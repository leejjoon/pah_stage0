# COM-centred copy of a D1 MLFF dataset for the charge-constrained dipole model (AtomicDielectricMACE,
# d1/mace_torch_dielectric.yaml). That model predicts mu = sum_i q_i r_i + sum_i p_i with sum_i q_i = q from raw positions,
# so for an ion (q != 0) it is not translation invariant: q * R shifts with the origin. The label dft_dipole is
# mu - q R_COM (d1/dft_seed_run.py, isotopic masses as g16_io.MASS), i.e. the dipole about the centre of mass. Putting
# each geometry's COM at the origin makes model and label refer to the same point; inference must centre the same way
# (d1/mlff_dipole_eval.py does).
#   python d1/mlff_dipole_prep.py <dataset dir> <out dir> [--delta-xtb]
# --delta-xtb: also add info key dipole_minus_xtb = dft_dipole - xtb_dipole (e*A, both about the COM; xtb_dipole from
# d1/mlff_qa.py, GFN2-xTB), the label of the Delta-xTB dipole model (d1/mace_torch_dipole_dxtb.yaml); the residual has
# zero net charge, so models of it need no total_charge.
# Rewrites only the three position columns of every frame (text level, so every info field, the forces and the
# Properties header stay byte-identical, apart from the added dipole_minus_xtb key). total_charge comes from the existing `charge` info key at training time
# (--total_charge_key charge); polarizability labels are not needed (mace-torch fills zeros, and the loss weight is 0).
# Checks per frame: |COM| < 1e-6 A after centring, interatomic distances unchanged, and the uniform-charge identity
# q * mean(r') = q (R_geo - R_COM) evaluated on the raw geometry.
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g16_io import MASS  # noqa: E402  (same isotopic masses as the dft_dipole label)


def centre_file(src, dst, delta_xtb=False):
    lines = open(src).read().splitlines()
    out, i, n_frames, worst = [], 0, 0, 0.0
    while i < len(lines):
        n = int(lines[i]); hdr = lines[i + 1]; rows = [l.split() for l in lines[i + 2:i + 2 + n]]
        assert hdr.startswith("Properties=species:S:1:pos:R:3"), f"{src}: unexpected header {hdr[:60]}"
        q = float(hdr.split(" charge=", 1)[1].split()[0])
        sym = [r[0] for r in rows]
        pos = np.array([[float(x) for x in r[1:4]] for r in rows])
        m = np.array([MASS[s] for s in sym])
        com = (m[:, None] * pos).sum(0) / m.sum()
        new = pos - com
        com_new = (m[:, None] * new).sum(0) / m.sum()
        d_old = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
        d_new = np.linalg.norm(new[:, None] - new[None], axis=-1)
        uni = q * new.mean(0) - q * (pos.mean(0) - com)
        worst = max(worst, np.abs(com_new).max(), np.abs(d_new - d_old).max(), np.abs(uni).max())
        if delta_xtb:
            vec = lambda k: np.array([float(x) for x in hdr.split(f' {k}="', 1)[1].split('"', 1)[0].split()])
            d = vec("dft_dipole") - vec("xtb_dipole")
            hdr = hdr.replace(' dft_dipole="', ' dipole_minus_xtb="' + " ".join(f"{x:.6f}" for x in d) + '" dft_dipole="', 1)
        out += [lines[i], hdr]
        for r, p in zip(rows, new):
            out.append(" ".join([r[0]] + [f"{x:.8f}" for x in p] + r[4:]))
        i += 2 + n; n_frames += 1
    assert worst < 1e-6, f"{src}: centring check failed ({worst:.2e})"
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")
    return n_frames, worst


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("out")
    ap.add_argument("--delta-xtb", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.dataset, "*.extxyz")))
    assert files, f"no extxyz in {a.dataset}"
    for f in files:
        n, w = centre_file(f, os.path.join(a.out, os.path.basename(f)), a.delta_xtb)
        print(f"{os.path.basename(f):26s} {n:5d} frames  max check residual {w:.1e}")


if __name__ == "__main__":
    main()
