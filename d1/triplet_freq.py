# Recompute a PAHdb v4.00 non-Kekule species in its high-spin state: UKS B3LYP/6-31G* (6D, (99,590) grid, conv_tol 1e-11,
# as PAHdb and d1/pyscf_freq_fd.py), optional geometry optimisation (GPU4PySCF's ASE BFGS), then analytic harmonic
# frequencies and IR intensities (d1/uks_ir.py). Multiplicity = |sublattice imbalance| + 1 (Lieb), from
# d1/pahdb_v4_exclude_nonkekule_v1.csv unless given.
#   python triplet_freq.py <pahdb v4.00 xml.gz> <uid> <out.json> [--opt] [--fmax 0.02] [--mult M]
# out.json: uid, formula, mult, charge, energies, <S^2>, optimisation steps and convergence, timings, the geometry used
# (Angstrom), frequencies (cm^-1, unscaled) and intensities (km/mol).
import argparse, csv, gzip, json, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYM = {1: "H", 6: "C", 7: "N"}


def pahdb_geom(xml, uid):
    with gzip.open(xml) as f:
        for _, sp in ET.iterparse(f, events=("end",)):
            if sp.tag == NS + "specie" and int(sp.get("uid")) == uid:
                at = sp.find(NS + "geometry").findall(NS + "atom")
                return ([SYM[int(a.find(NS + "type").text)] for a in at],
                        [[float(a.find(NS + t).text) for t in ("x", "y", "z")] for a in at],
                        int(sp.find(NS + "charge").text), sp.find(NS + "formula").text)
            if sp.tag == NS + "specie":
                sp.clear()


def build(sym, pos, charge, mult):
    from pyscf import gto
    from gpu4pyscf import dft
    mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", charge=charge, spin=mult - 1, unit="Angstrom", cart=True, verbose=3)
    mf = dft.UKS(mol); mf.xc = "b3lypg"; mf.conv_tol = 1e-11; mf.max_cycle = 200; mf.grids.atom_grid = (99, 590)
    return mol, mf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("xml"); ap.add_argument("uid", type=int); ap.add_argument("out")
    ap.add_argument("--opt", action="store_true"); ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--mult", type=int, default=None); ap.add_argument("--max-steps", type=int, default=150)
    a = ap.parse_args()
    from uks_ir import eval_ir_freq_intensity_uks
    sym, pos, charge, formula = pahdb_geom(a.xml, a.uid)
    mult = a.mult
    if mult is None:
        lst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pahdb_v4_exclude_nonkekule_v1.csv")
        imb = {int(r["uid"]): abs(int(r["sublattice_imbalance"])) for r in csv.DictReader(open(lst))}
        mult = imb[a.uid] + 1
    res = dict(uid=a.uid, formula=formula, charge=charge, mult=mult, natom=len(sym))
    T0 = time.time()
    mol, mf = build(sym, pos, charge, mult)
    t = time.time(); e0 = mf.kernel(); res.update(E_pahdb_geom=float(e0), t_scf=round(time.time() - t, 1), converged_scf=bool(mf.converged))
    if a.opt:
        from gpu4pyscf.geomopt.ase_solver import kernel as ase_opt
        steps = []
        t = time.time()
        conv, mol = ase_opt(mf, fmax=a.fmax, max_steps=a.max_steps, logfile=a.out + ".opt.log")
        res.update(opt_converged=bool(conv), t_opt=round(time.time() - t, 1),
                   opt_steps=sum(1 for l in open(a.out + ".opt.log") if l.startswith("BFGS:")) - 1)
        pos = (mol.atom_coords() * 0.529177210903).tolist()
        mol, mf = build(sym, pos, charge, mult)
        mf.kernel()
        res.update(E_opt=float(mf.e_tot), dE_opt_kcal=round((mf.e_tot - e0) * 627.5095, 3))
    res.update(S2=float(mf.spin_square()[0]))
    t = time.time(); freq, inten, _ = eval_ir_freq_intensity_uks(mf, mf.Hessian()); res.update(t_hess_ir=round(time.time() - t, 1))
    f = np.asarray(freq); I = np.asarray(inten)
    res.update(n_imag=int((f < 0).sum()), lowest=round(float(f.min()), 1), total=round(float(I.sum()), 1),
               ch_share=round(float(I[f > 2700].sum() / I.sum()), 3), t_total=round(time.time() - T0, 1),
               nao=int(mol.nao), sym=sym, pos=pos, freq=f.tolist(), I=I.tolist())
    json.dump(res, open(a.out, "w"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("sym", "pos", "freq", "I")}, indent=1))


if __name__ == "__main__":
    main()
