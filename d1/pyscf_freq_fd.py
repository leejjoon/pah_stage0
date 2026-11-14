# Harmonic frequencies and IR intensities with PySCF by central finite differences of analytic gradients and dipoles,
# to compare with PAHdb's Gaussian B3LYP/6-31G* values at the same (PAHdb-optimised) geometry. With no Gaussian
# runs of our own (no Gaussian access on Hangang, 2026-11-13), this is the test of forces (Hessian = their
# derivative) and dipoles (intensities = dipole derivatives) against Gaussian. Settings as T1 of
# docs/plan/handoff_pyscf_gpu_jax.md: b3lypg, 6D d (cart), (99, 590) grid, conv_tol 1e-11.
#   python pyscf_freq_fd.py run <geom.gjf|.xyz> <workdir> --charge 1 --mult 2 [--part 0 --nparts 2] [--h 0.005]
#   python pyscf_freq_fd.py analyse <geom> <workdir> --uid 359 --pahdb <pahdb.xml[.gz]>
# run: each displaced point (atom, axis, sign; step h bohr) is written as <workdir>/d_<i>_<x>_<s>.npz; several
# parts can run at once on different cores. analyse: Hessian (symmetrised), normal modes with Gaussian's masses
# (most abundant isotope), intensities in km/mol, and a mode-by-mode comparison with the PAHdb entry.
import argparse, gzip, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyscf_match import read_geom

BOHR = 0.529177210903


def build(sym, pos_bohr, charge, mult, grid=(99, 590), tol=1e-11):
    from pyscf import gto, dft
    mol = gto.M(atom=list(zip(sym, pos_bohr)), basis="6-31g*", charge=charge, spin=mult - 1, unit="Bohr",
                cart=True, verbose=0)
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol); mf.xc = "b3lypg"; mf.conv_tol = tol; mf.max_cycle = 200
    mf.grids.atom_grid = grid
    return mol, mf


def point(sym, P, charge, mult, dm0=None):
    mol, mf = build(sym, P, charge, mult)
    e = mf.kernel(dm0=dm0)
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ)   # newton().kernel takes MOs first, not a density
    g = mf.nuc_grad_method().kernel()
    mu = np.asarray(mf.dip_moment(unit="AU", verbose=0))          # about the origin; derivatives are origin-free
    return e, g, mu, mf.make_rdm1(), mf.converged


def run(a):
    sym, pos = read_geom(a.geom); P0 = pos / BOHR; n = len(sym)
    os.makedirs(a.workdir, exist_ok=True)
    f0 = f"{a.workdir}/eq.npz"
    if a.part == 0 and not os.path.exists(f0):
        e, g, mu, dm, ok = point(sym, P0, a.charge, a.mult)
        np.savez(f0, e=e, g=g, mu=mu, ok=ok, P=P0)
    _, _, _, dm0, _ = point(sym, P0, a.charge, a.mult) if a.reuse_dm else (None,) * 5
    jobs = [(i, x, s) for i in range(n) for x in range(3) for s in (1, -1)]
    for k, (i, x, s) in enumerate(jobs):
        if k % a.nparts != a.part:
            continue
        f = f"{a.workdir}/d_{i}_{x}_{'p' if s > 0 else 'm'}.npz"
        if os.path.exists(f):
            continue
        P = P0.copy(); P[i, x] += s * a.h
        t = time.time(); e, g, mu, _, ok = point(sym, P, a.charge, a.mult, dm0)
        np.savez(f, e=e, g=g, mu=mu, ok=ok)
        print(f"{k + 1}/{len(jobs)} atom {i} axis {x} {s:+d} {time.time() - t:.1f}s conv={ok}", flush=True)


def pahdb_modes(path, uid):
    op = gzip.open if path.endswith(".gz") else open
    buf, on = [], False
    for line in op(path, "rt"):
        if f'<specie uid="{uid}"' in line:
            on = True
        if on:
            buf.append(line)
            if "</specie>" in line:
                break
    s = "".join(buf)
    # PAHdb v4 stores *scaled* frequencies, each with its own scale attribute (0.9794 / 0.9691 / 0.9597 by range);
    # divide it out to get Gaussian's harmonic frequency.
    modes = re.findall(r'<frequency scale="([\d.]+)">([-\d.]+)</frequency>\s*<intensity>([-\d.]+)</intensity>\s*'
                       r'<symmetry>(\w+)</symmetry>', s)
    return (np.array([[float(f) / float(sc), float(i), float(sc)] for sc, f, i, _ in modes]), [m[3] for m in modes],
            float(re.search(r"<total_e>([-\d.]+)", s).group(1)))


def analyse(a):
    from pyscf import gto
    from pyscf.hessian import thermo
    from pyscf.data import nist
    sym, pos = read_geom(a.geom); n = len(sym)
    H = np.zeros((n, 3, n, 3)); dmu = np.zeros((n, 3, 3)); bad = []
    for i in range(n):
        for x in range(3):
            p = np.load(f"{a.workdir}/d_{i}_{x}_p.npz"); m = np.load(f"{a.workdir}/d_{i}_{x}_m.npz")
            if not (p["ok"] and m["ok"]):
                bad.append((i, x))
            H[i, x] = (p["g"] - m["g"]) / (2 * a.h)
            dmu[i, x] = (p["mu"] - m["mu"]) / (2 * a.h)
    H = H.transpose(0, 2, 1, 3)                                  # (atom i, atom j, xi, xj) as thermo expects
    mol = gto.M(atom=list(zip(sym, pos)), unit="Angstrom", basis="sto-3g", charge=a.charge, spin=a.mult - 1, verbose=0)
    mass = mol.atom_mass_list(isotope_avg=False)                 # Gaussian: most abundant isotope
    r = thermo.harmonic_analysis(mol, H, mass=mass, imaginary_freq=False)
    freq = np.asarray(r["freq_wavenumber"]); nm = r["norm_mode"]  # nm: bohr per (amu^1/2 bohr)
    # An ion's dipole depends on the origin, but normal modes keep the centre of mass fixed, so dmu/dQ about the
    # coordinate origin equals dmu/dQ about the centre of mass (mu_COM = mu_origin - q R_COM, R_COM constant).
    dmudQ = np.einsum("ixc,kix->kc", dmu, nm)                    # e*bohr / (amu^1/2 bohr) = e / amu^1/2
    au = nist.E_CHARGE**2 / nist.ATOMIC_MASS                     # (C/kg^1/2)^2
    NA = nist.AVOGADRO if hasattr(nist, "AVOGADRO") else 6.02214076e23
    eps0, c = 8.8541878128e-12, nist.LIGHT_SPEED_SI
    inten = NA / (12 * eps0 * c**2) * au * (dmudQ**2).sum(1) / 1e3          # km/mol
    ref, symm, e_ref = pahdb_modes(a.pahdb, a.uid)
    eq = np.load(f"{a.workdir}/eq.npz")
    o = np.argsort(freq); freq, inten = freq[o], inten[o]
    df = freq - ref[:, 0]; strong = ref[:, 1] > 1.0
    di = inten - ref[:, 1]
    out = dict(uid=a.uid, nmodes=len(freq), energy=float(eq["e"]), energy_pahdb=e_ref,
               dE_uHa=(float(eq["e"]) - e_ref) * 1e6, fmax_eq=float(np.abs(eq["g"]).max()), unconverged=bad,
               freq_mad=float(np.abs(df).mean()), freq_maxabs=float(np.abs(df).max()),
               freq_max_mode=int(np.abs(df).argmax()),
               int_rel_mad_strong=float(np.abs(di[strong] / ref[strong, 1]).mean()),
               int_rel_max_strong=float(np.abs(di[strong] / ref[strong, 1]).max()),
               int_abs_max=float(np.abs(di).max()), total_int=float(inten.sum()), total_int_pahdb=float(ref[:, 1].sum()),
               freq=freq.tolist(), inten=inten.tolist(), freq_pahdb_unscaled=ref[:, 0].tolist(), scale_pahdb=ref[:, 2].tolist(), inten_pahdb=ref[:, 1].tolist(),
               sym_pahdb=symm, h_bohr=a.h, date=time.strftime("%Y-%m-%d"))
    json.dump(out, open(f"{a.workdir}/compare.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list) or k == "unconverged"}, indent=1))
    print(" mode  sym   f_pyscf  f_pahdb/scale  df    scale    I_pyscf   I_pahdb")
    for k in range(len(freq)):
        print(f"{k:4d}  {symm[k]:4s} {freq[k]:9.2f} {ref[k, 0]:9.2f} {df[k]:+6.2f}  {ref[k, 2]:.4f} {inten[k]:9.3f} {ref[k, 1]:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyse"]); ap.add_argument("geom"); ap.add_argument("workdir")
    ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--h", type=float, default=0.005, help="step (bohr)")
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reuse-dm", action="store_true", help="start each SCF from the equilibrium density")
    ap.add_argument("--uid", type=int); ap.add_argument("--pahdb")
    a = ap.parse_args()
    from pyscf import lib
    lib.num_threads(a.threads)
    run(a) if a.cmd == "run" else analyse(a)


if __name__ == "__main__":
    main()
