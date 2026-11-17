# Test of d1/uks_ir.py (analytic open-shell IR) on one GPU, at PAHdb v4.00 geometries, B3LYP/6-31G* as PAHdb and
# d1/pyscf_freq_fd.py (b3lypg, 6D, (99,590) grid, conv_tol 1e-11).
#   python uks_ir_test.py <pahdb v4.00 xml.gz> <test> <out json>
# tests:
#   closed  naphthalene (uid 330) as RKS with GPU4PySCF's own eval_ir_freq_intensity and as UKS (spin 0) with
#           eval_ir_freq_intensity_uks: the two must agree (checks the open-shell code against the closed-shell one);
#   cation  naphthalene cation (uid 359), UKS doublet, against PAHdb's listed spectrum (frequencies unscaled with
#           each mode's scale attribute);
#   triplet C30H14 uid 10770 (no Kekule structure, listed as a singlet): RKS singlet (should reproduce PAHdb's
#           inflated intensities) and UKS triplet (expected normal), with <S^2>, energies, timings.
import gzip, json, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
SYM = {1: "H", 6: "C"}


def pahdb(xml, uid):
    with gzip.open(xml) as f:
        for _, sp in ET.iterparse(f, events=("end",)):
            if sp.tag == NS + "specie" and int(sp.get("uid")) == uid:
                at = sp.find(NS + "geometry").findall(NS + "atom")
                sym = [SYM[int(a.find(NS + "type").text)] for a in at]
                pos = [[float(a.find(NS + t).text) for t in ("x", "y", "z")] for a in at]
                ms = sp.find(NS + "transitions").findall(NS + "mode")
                nu = [float(m.find(NS + "frequency").text) / float(m.find(NS + "frequency").get("scale", 1)) for m in ms]
                I = [float(m.find(NS + "intensity").text) for m in ms]
                return dict(sym=sym, pos=pos, nu=nu, I=I, charge=int(sp.find(NS + "charge").text), state=sp.find(NS + "symmetry").text)
            if sp.tag == NS + "specie":
                sp.clear()


def build(sym, pos, charge, mult):
    from pyscf import gto
    from gpu4pyscf import dft
    mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", charge=charge, spin=mult - 1, unit="Angstrom", cart=True, verbose=3)
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol); mf.xc = "b3lypg"; mf.conv_tol = 1e-11; mf.max_cycle = 200
    mf.grids.atom_grid = (99, 590)
    return mol, mf


def run(sym, pos, charge, mult, closed_shell_code=False, force_uks=False):
    from gpu4pyscf import dft
    from gpu4pyscf.properties.ir import eval_ir_freq_intensity
    from uks_ir import eval_ir_freq_intensity_uks
    mol, mf = build(sym, pos, charge, mult)
    if force_uks:
        mf = dft.UKS(mol); mf.xc = "b3lypg"; mf.conv_tol = 1e-11; mf.max_cycle = 200; mf.grids.atom_grid = (99, 590)
    t = time.time(); e = mf.kernel(); t_scf = time.time() - t
    s2 = float(mf.spin_square()[0]) if (mult > 1 or force_uks) else 0.0   # <S^2>: 0 singlet, 0.75 doublet, 2 triplet
    t = time.time()
    h = mf.Hessian()
    if closed_shell_code:
        freq, inten = eval_ir_freq_intensity(mf, h)
        freq, inten = np.asarray(freq.get() if hasattr(freq, "get") else freq), np.asarray(inten.get() if hasattr(inten, "get") else inten)
    else:
        freq, inten, _ = eval_ir_freq_intensity_uks(mf, h)
    t_hess = time.time() - t
    return dict(E=float(e), converged=bool(mf.converged), S2=s2, t_scf=round(t_scf, 1), t_hess_ir=round(t_hess, 1),
                nao=int(mol.nao), freq=[float(x) for x in freq], I=[float(x) for x in inten])


def summary(r, ref=None):
    f = np.array(r["freq"]); I = np.array(r["I"])
    out = dict(total=round(float(I.sum()), 1), ch_share=round(float(I[f > 2700].sum() / I.sum()), 3),
               n_imag=int((f < 0).sum()), lowest=round(float(f.min()), 1))
    if ref is not None:
        rf = np.sort(np.array(ref["nu"])); ff = np.sort(f[f > 0])[-len(rf):]
        ri = np.array(ref["I"])[np.argsort(ref["nu"])]; fi = I[f > 0][np.argsort(f[f > 0])][-len(rf):]
        strong = ri > 1
        out.update(ref_total=round(float(ri.sum()), 1), d_freq_median=round(float(np.median(ff - rf)), 2),
                   strong_rel_I_mean=round(float(np.mean(np.abs(fi[strong] - ri[strong]) / ri[strong])), 3))
    return out


def main(xml, test, outp):
    res = {}
    if test == "closed":
        m = pahdb(xml, 330)
        a = run(m["sym"], m["pos"], 0, 1, closed_shell_code=True); b = run(m["sym"], m["pos"], 0, 1, force_uks=True)
        fa, fb = np.array(a["freq"]), np.array(b["freq"]); Ia, Ib = np.array(a["I"]), np.array(b["I"])
        res = dict(rks_builtin=summary(a, m), uks_ours=summary(b, m), max_d_freq=float(np.abs(fa - fb).max()),
                   max_d_I=float(np.abs(Ia - Ib).max()), max_rel_d_I_strong=float(np.max(np.abs(Ia - Ib)[Ia > 1] / Ia[Ia > 1])),
                   timings=dict(rks=[a["t_scf"], a["t_hess_ir"]], uks=[b["t_scf"], b["t_hess_ir"]]))
    elif test == "cation":
        m = pahdb(xml, 359)
        r = run(m["sym"], m["pos"], 1, 2)
        res = dict(uks_doublet=summary(r, m), S2=r["S2"], timings=[r["t_scf"], r["t_hess_ir"]], pahdb_state=m["state"])
    elif test == "triplet":
        m = pahdb(xml, 10770)
        s = run(m["sym"], m["pos"], 0, 1, closed_shell_code=True); t = run(m["sym"], m["pos"], 0, 3)
        res = dict(pahdb=dict(state=m["state"], total=round(sum(m["I"]), 1),
                              ch_share=round(sum(i for n, i in zip(m["nu"], m["I"]) if n > 2700) / sum(m["I"]), 3)),
                   rks_singlet=dict(summary(s), E=s["E"], timings=[s["t_scf"], s["t_hess_ir"]], nao=s["nao"]),
                   uks_triplet=dict(summary(t), E=t["E"], S2=t["S2"], timings=[t["t_scf"], t["t_hess_ir"]]),
                   E_triplet_minus_singlet_kcal=round((t["E"] - s["E"]) * 627.5095, 2))
        res["spectra"] = dict(singlet=dict(freq=s["freq"], I=s["I"]), triplet=dict(freq=t["freq"], I=t["I"]))
    json.dump(res, open(outp, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "spectra"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
