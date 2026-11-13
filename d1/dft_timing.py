# DFT single-point timing for planning charged-state MLFF training data (docs/plan/charged_tdep_validation.md):
# energy, forces and dipole at B3LYP/4-31G (the PAHdb and Mai et al. 2025 level), restricted for closed shells,
# unrestricted for doublets, on a thermally distorted geometry (as in training data, where SCF converges slower than
# at the minimum). PySCF's "b3lypg" is Gaussian's B3LYP (VWN3), so the data match Gaussian-built sets.
#   python dft_timing.py <xyz> --charge 1 --mult 2 [--displace 0.05] [--seed 0] [--threads 8] [--out result.json]
# Prints and optionally writes: atoms, electrons, basis functions, SCF cycles and time, gradient time, <S^2>,
# energy, |dipole| about the centre of mass.
import argparse, json, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz"); ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--displace", type=float, default=0.05, help="Gaussian displacement per coordinate (A)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--basis", default="4-31g"); ap.add_argument("--xc", default="b3lypg")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import numpy as np
    from pyscf import gto, dft, lib
    if a.threads:
        lib.num_threads(a.threads)
    lines = open(a.xyz).read().split("\n"); n = int(lines[0])
    sym = [l.split()[0] for l in lines[2:2 + n]]
    pos = np.array([[float(x) for x in l.split()[1:4]] for l in lines[2:2 + n]])
    pos = pos + np.random.default_rng(a.seed).normal(0, a.displace, pos.shape)
    mol = gto.M(atom=list(zip(sym, pos)), basis=a.basis, charge=a.charge, spin=a.mult - 1, unit="Angstrom", verbose=0)
    mf = (dft.RKS if a.mult == 1 else dft.UKS)(mol)
    if os.environ.get("DF"):                                # DF=1: density fitting (faster; small error vs Gaussian)
        mf = mf.density_fit()
    mf.xc = a.xc; mf.conv_tol = 1e-9; mf.max_cycle = 200
    cycles = []
    mf.callback = lambda env: cycles.append(env["cycle"])
    t0 = time.time(); e = mf.kernel(); t_scf = time.time() - t0
    if not mf.converged:                                    # second-order fallback, as production would use
        mf = mf.newton(); t1 = time.time(); e = mf.kernel(mf.mo_coeff, mf.mo_occ); t_scf += time.time() - t1  # newton().kernel takes MOs first, not a density
    t0 = time.time(); g = mf.nuc_grad_method().kernel(); t_grad = time.time() - t0
    m = mol.atom_mass_list(); com = (m[:, None] * mol.atom_coords(unit="Angstrom")).sum(0) / m.sum()
    # PySCF's dipole is about the coordinate origin; for an ion shift it to the centre of mass (4.80321 Debye per e*A)
    mu = np.asarray(mf.dip_moment(unit="Debye", verbose=0)) - a.charge * com * 4.80321
    s2 = float(mf.spin_square()[0]) if a.mult > 1 else 0.0
    r = dict(xyz=os.path.basename(a.xyz), charge=a.charge, mult=a.mult, atoms=n, electrons=int(mol.nelectron),
             nao=int(mol.nao), threads=lib.num_threads(), converged=bool(mf.converged), scf_cycles=len(cycles),
             t_scf=round(t_scf, 2), t_grad=round(t_grad, 2), t_total=round(t_scf + t_grad, 2), s2=round(s2, 4),
             energy=float(e), dipole_debye=round(float(np.linalg.norm(mu)), 4), fmax=float(np.abs(g).max()),
             xc=a.xc, basis=a.basis, density_fit=bool(os.environ.get("DF")))
    print(json.dumps(r))
    if a.out:
        json.dump(r, open(a.out, "w"))


if __name__ == "__main__":
    main()
