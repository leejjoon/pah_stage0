# Recompute a PAHdb v4.00 non-Kekule species in its high-spin state: UKS B3LYP/6-31G* (6D, (99,590) grid, conv_tol 1e-11,
# as PAHdb and d1/pyscf_freq_fd.py), optional geometry optimisation (ASE BFGS), then analytic harmonic frequencies and
# IR intensities (d1/uks_ir.py). Multiplicity = |sublattice imbalance| + 1 (Lieb), from
# d1/pahdb_v4_exclude_nonkekule_v1.csv unless given.
#   python triplet_freq.py <pahdb v4.00 xml.gz> <uid> <out.json> [--opt] [--fmax 0.02] [--mult M] [--scf-only]
# Every SCF starts from the previous density, falls back to second-order (Newton) SCF from the current orbitals when
# DIIS does not converge (as d1/aimd_pyscf.py), and the run stops if it still has not converged (job 1294335: one SCF of
# the C30H14 optimisation and the C60H20 SCF failed; the old driver went on to the Hessian).
# Run with CUPY_ACCELERATORS=cub: container D's default cub,cutensor routes elementwise CuPy operations through cuTENSOR,
# which fails in the UHF CPHF for C84 and larger (CUTENSOR_STATUS_NOT_SUPPORTED, job 1294335).
# out.json: uid, formula, mult, energies, <S^2>, SCF fallbacks, optimisation steps, timings, geometry (Angstrom),
# frequencies (cm^-1, unscaled) and intensities (km/mol).
import argparse, csv, gzip, json, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYM = {1: "H", 6: "C", 7: "N"}
HARTREE_EV, BOHR_A = 27.211386245988, 0.529177210903


def pahdb_geom(xml, uid):
    with gzip.open(xml) as f:
        for _, sp in ET.iterparse(f, events=("end",)):
            if sp.tag == NS + "specie" and int(sp.get("uid")) == uid:
                at = sp.find(NS + "geometry").findall(NS + "atom")
                return ([SYM[int(a.find(NS + "type").text)] for a in at],
                        np.array([[float(a.find(NS + t).text) for t in ("x", "y", "z")] for a in at]),
                        int(sp.find(NS + "charge").text), sp.find(NS + "formula").text)
            if sp.tag == NS + "specie":
                sp.clear()


class SCF:
    """UKS at a geometry, density reused, Newton fallback, hard stop if unconverged."""

    def __init__(self, sym, charge, mult):
        self.sym, self.q, self.m = sym, charge, mult; self.dm = None; self.fallbacks = 0; self.n = 0

    def run(self, pos):
        from pyscf import gto
        from gpu4pyscf import dft
        mol = gto.M(atom=list(zip(self.sym, pos)), basis="6-31g*", charge=self.q, spin=self.m - 1, unit="Angstrom",
                    cart=True, verbose=3)
        mf = dft.UKS(mol); mf.xc = "b3lypg"; mf.conv_tol = 1e-11; mf.max_cycle = 200; mf.grids.atom_grid = (99, 590)
        mf.kernel(dm0=self.dm); self.n += 1
        if not mf.converged:
            self.fallbacks += 1
            mf = mf.newton(); mf.kernel(mf.mo_coeff, mf.mo_occ)      # newton().kernel takes MOs first, not a density
            if not mf.converged:
                raise RuntimeError(f"SCF not converged even with Newton (SCF call {self.n})")
        self.dm = mf.make_rdm1()
        return mf


def optimise(scf, pos, fmax, max_steps, logfile):
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.calculators.calculator import Calculator, all_changes

    class Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            mf = scf.run(self.atoms.positions)
            g = mf.nuc_grad_method().kernel()
            self.results = {"energy": float(mf.e_tot) * HARTREE_EV, "forces": -np.asarray(g.get() if hasattr(g, "get") else g) * HARTREE_EV / BOHR_A}
    at = Atoms(scf.sym, positions=pos); at.calc = Calc()
    opt = BFGS(at, logfile=logfile)
    conv = opt.run(fmax=fmax, steps=max_steps)
    return bool(conv), at.positions.copy(), opt.nsteps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("xml"); ap.add_argument("uid", type=int); ap.add_argument("out")
    ap.add_argument("--opt", action="store_true"); ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--mult", type=int, default=None); ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--scf-only", action="store_true", help="stop after the SCF at the PAHdb geometry (convergence check)")
    a = ap.parse_args()
    sym, pos, charge, formula = pahdb_geom(a.xml, a.uid)
    mult = a.mult
    if mult is None:
        lst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pahdb_v4_exclude_nonkekule_v1.csv")
        mult = {int(r["uid"]): abs(int(r["sublattice_imbalance"])) for r in csv.DictReader(open(lst))}[a.uid] + 1
    res = dict(uid=a.uid, formula=formula, charge=charge, mult=mult, natom=len(sym),
               cupy_accelerators=os.environ.get("CUPY_ACCELERATORS"))
    T0 = time.time(); scf = SCF(sym, charge, mult)
    t = time.time(); mf = scf.run(pos)
    res.update(E_pahdb_geom=float(mf.e_tot), t_scf=round(time.time() - t, 1), S2_pahdb_geom=float(mf.spin_square()[0]))
    if not a.scf_only:
        if a.opt:
            t = time.time()
            conv, pos, nsteps = optimise(scf, pos, a.fmax, a.max_steps, a.out + ".opt.log")
            if not conv:
                raise RuntimeError(f"optimisation not converged in {nsteps} steps")
            mf = scf.run(pos)
            res.update(opt_steps=int(nsteps), t_opt=round(time.time() - t, 1), E_opt=float(mf.e_tot),
                       dE_opt_kcal=round((mf.e_tot - res["E_pahdb_geom"]) * 627.5095, 3))
        from uks_ir import eval_ir_freq_intensity_uks
        t = time.time(); freq, inten, _ = eval_ir_freq_intensity_uks(mf, mf.Hessian()); res.update(t_hess_ir=round(time.time() - t, 1))
        f = np.asarray(freq); I = np.asarray(inten)
        res.update(n_imag=int((f < 0).sum()), lowest=round(float(f.min()), 1), total=round(float(I.sum()), 1),
                   ch_share=round(float(I[f > 2700].sum() / I.sum()), 3), freq=f.tolist(), I=I.tolist())
    res.update(S2=float(mf.spin_square()[0]), scf_calls=scf.n, scf_newton_fallbacks=scf.fallbacks, nao=int(mf.mol.nao),
               t_total=round(time.time() - T0, 1), sym=sym, pos=np.asarray(pos).tolist())
    json.dump(res, open(a.out, "w"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("sym", "pos", "freq", "I")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
