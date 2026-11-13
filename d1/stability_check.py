# Stability analysis (handoff §2: Gaussian's Stable=Opt for UKS, RKS->UKS test for closed shells) with GPU4PySCF
# against CPU PySCF, on the same displaced geometry and the T1 settings (b3lypg, 6-31G* 6D, (99, 590), conv_tol 1e-9,
# conv_tol_grad 1e-6). Reports stable / not stable, the energy after any re-optimisation, and the time.
#   python stability_check.py <geom> --charge 1 --mult 2 [--gpu] [--displace 0.05 --seed 0] [--out r.json]
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyscf_match import read_geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geom"); ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--gpu", action="store_true"); ap.add_argument("--displace", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    from pyscf import gto, dft, lib
    lib.num_threads(a.threads)
    sym, pos = read_geom(a.geom)
    if a.displace:
        pos = pos + np.random.default_rng(a.seed).normal(0, a.displace, pos.shape)
    mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", charge=a.charge, spin=a.mult - 1, unit="Angstrom",
                cart=True, verbose=0)
    mf = (dft.RKS if a.mult == 1 else dft.UKS)(mol); mf.xc = "b3lypg"; mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200
    if a.gpu:
        mf = mf.to_gpu()
    t0 = time.time(); e0 = mf.kernel(); t_scf = time.time() - t0
    r = dict(geom=os.path.basename(a.geom), charge=a.charge, mult=a.mult, gpu=a.gpu, e_scf=float(e0), t_scf=round(t_scf, 1))
    t0 = time.time()
    if a.mult == 1:      # RKS: internal and RKS->UKS (external, test only, as Gaussian's Stable)
        _, _, st_i, st_e = mf.stability(internal=True, external=True, return_status=True)
        r.update(stable_internal=bool(st_i), stable_external=bool(st_e), t_stab=round(time.time() - t0, 1))
    else:                # UKS: internal, re-optimised until stable (Gaussian's Stable=Opt), at most 5 rounds
        n = 0
        while True:
            mo, _, st, _ = mf.stability(internal=True, external=False, return_status=True)
            if st or n == 5:
                break
            n += 1
            mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
        r.update(stable_internal=bool(st), reopt_rounds=n, e_final=float(mf.e_tot), t_stab=round(time.time() - t0, 1),
                 s2=float(mf.spin_square()[0]))
    r.update(date=time.strftime("%Y-%m-%d"), host=os.uname().nodename)
    print(json.dumps(r), flush=True)
    if a.out:
        json.dump(r, open(a.out, "w"))


if __name__ == "__main__":
    main()
