# T1 of docs/plan/handoff_pyscf_gpu_jax.md: which PySCF settings reproduce Gaussian's B3LYP/6-31G* (energies, and
# later forces and dipoles from the Phase 0 logs). One single point (energy, analytic forces, dipole about the centre
# of mass, <S^2>) per named setting, on a fixed geometry; each result is written as JSON.
# Settings vary the integration grid (size, pruning, radial scheme, Becke partition), 5D vs 6D d functions and
# density fitting; all use Gaussian's B3LYP ("b3lypg", VWN3) and a tight SCF (conv_tol 1e-11 Ha).
#   python pyscf_match.py <geom.xyz|.gjf> <outdir> --charge 1 --mult 2 --tag naph_q1 [--threads 8] [--only a,b]
#                         [--displace 0.05 --seed 0] [--gpu]
# Geometry: an .xyz, or a Gaussian .gjf (Cartesian block after the charge/multiplicity line, as written by
# g16_phase0.py), so the PAHdb coordinates are used at their full 8 decimals.
import argparse, json, os, time
import numpy as np


def mhl(n, charge, *args, **kwargs):
    """Euler-Maclaurin radial grid of Murray, Handy and Laming (Mol. Phys. 78, 997, 1993), the scheme Gaussian's
    grids are built on: r = R x^2/(1-x)^2, x = i/(n+1). R is the Bragg radius here; Gaussian's exact radii are not
    published, so this is Gaussian-like, not Gaussian's grid."""
    from pyscf.dft import radi
    R = radi.BRAGG_RADII[charge]
    x = np.arange(1, n + 1) / (n + 1)
    return R * x**2 / (1 - x)**2, 2 * R * x / (1 - x)**3 / (n + 1)


# name: (atom_grid or level, prune, radial, becke partition, cart, density fit auxbasis, small_rho_cutoff)
SETTINGS = {
    "L3":             (3, "nwchem", "ta", "becke", True, None, None),       # PySCF default grid
    "75x302":         ((75, 302), "nwchem", "ta", "becke", True, None, None),   # size of Gaussian's FineGrid
    "99x590":         ((99, 590), "nwchem", "ta", "becke", True, None, None),   # size of Gaussian's UltraFine
    "99x590_tol9":    ((99, 590), "nwchem", "ta", "becke", True, None, None),   # conv_tol 1e-9 (as aimd_pyscf.py)
    "99x590_t9g6":    ((99, 590), "nwchem", "ta", "becke", True, None, None),   # conv_tol 1e-9, conv_tol_grad 1e-6
    "99x590_noprune": ((99, 590), None, "ta", "becke", True, None, None),
    "99x590_sg1":     ((99, 590), "sg1", "ta", "becke", True, None, None),
    "99x590_ssf":     ((99, 590), "nwchem", "ta", "ssf", True, None, None),
    "99x590_mhl_ssf": ((99, 590), "nwchem", "mhl", "ssf", True, None, None),   # most Gaussian-like
    "75x302_mhl_ssf": ((75, 302), "nwchem", "mhl", "ssf", True, None, None),
    "99x590_5d":      ((99, 590), "nwchem", "ta", "becke", False, None, None),  # PySCF's default spherical d
    "99x590_df_jk":   ((99, 590), "nwchem", "ta", "becke", True, "def2-universal-jkfit", None),
    "99x590_df_def":  ((99, 590), "nwchem", "ta", "becke", True, "default", None),
    "ref_200x974":    ((200, 974), None, "ta", "becke", True, None, 0.0),      # grid-converged reference
}


def read_geom(path):
    lines = open(path).read().splitlines()
    if path.endswith(".gjf"):
        i = next(k for k, l in enumerate(lines) if len(l.split()) == 2 and all(t.lstrip("-").isdigit() for t in l.split()))
        block = []
        for l in lines[i + 1:]:
            if not l.strip():
                break
            block.append(l)
    else:
        block = lines[2:2 + int(lines[0])]
    return [l.split()[0] for l in block], np.array([[float(x) for x in l.split()[1:4]] for l in block])


def _np(x):
    return np.asarray(x.get() if hasattr(x, "get") else x)


def run(sym, pos, charge, mult, name, gpu=False):
    from pyscf import gto, dft
    from pyscf.dft import gen_grid, radi
    grid, prune, rad, part, cart, aux, rho_cut = SETTINGS[name]
    mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", charge=charge, spin=mult - 1, unit="Angstrom",
                cart=cart, verbose=0)
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol); mf.xc = "b3lypg"; mf.max_cycle = 200
    mf.conv_tol = 1e-9 if name.endswith(("_tol9", "_t9g6")) else 1e-11
    if name.endswith("_t9g6"):     # GPU4PySCF cannot reach 1e-11 for ~850+ basis functions (energy noise ~1e-9 Ha)
        mf.conv_tol_grad = 1e-6
    if isinstance(grid, int):
        mf.grids.level = grid
    else:
        mf.grids.atom_grid = grid
    mf.grids.prune = {"nwchem": gen_grid.nwchem_prune, "sg1": gen_grid.sg1_prune, None: None}[prune]
    mf.grids.radi_method = {"ta": radi.treutler_ahlrichs, "mhl": mhl}[rad]
    mf.grids.becke_scheme = {"becke": gen_grid.original_becke, "ssf": gen_grid.stratmann}[part]
    if rho_cut is not None:
        mf.small_rho_cutoff = rho_cut
    if aux:
        mf = mf.density_fit() if aux == "default" else mf.density_fit(auxbasis=aux)
    if gpu:                                                     # GPU4PySCF: same settings, carried over by to_gpu()
        mf = mf.to_gpu()
    n = [0]; mf.callback = lambda env: n.__setitem__(0, n[0] + 1)
    t0 = time.time(); e = mf.kernel(); t_scf = time.time() - t0
    newton = False
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ); newton = True; t_scf = time.time() - t0  # newton().kernel takes MOs first, not a density
    t0 = time.time(); g = _np(mf.nuc_grad_method().kernel()); t_grad = time.time() - t0
    m = mol.atom_mass_list(); com = (m[:, None] * mol.atom_coords(unit="Angstrom")).sum(0) / m.sum()
    mu = _np(mf.dip_moment(unit="Debye", verbose=0)) - charge * com * 4.80320471        # about the COM
    s2 = mf.spin_square() if mult > 1 else (0.0, 1.0)
    if gpu:
        import cupy
        mem = cupy.get_default_memory_pool().total_bytes() / 1e9
    return dict(setting=name + ("_gpu" if gpu else ""), energy=float(e), converged=bool(mf.converged), newton=newton,
                scf_cycles=n[0] or int(getattr(mf, "cycles", 0)), gpu_mem_gb=round(mem, 3) if gpu else None,
                nao=int(mol.nao), ngrids=int(mf.grids.weights.size), t_scf=round(t_scf, 2), t_grad=round(t_grad, 2),
                t_total=round(t_scf + t_grad, 2), s2=float(_np(s2[0])), grad=g.tolist(), fmax=float(np.abs(g).max()),
                dipole_debye=mu.tolist(), dipole_abs=float(np.linalg.norm(mu)), cart=cart, auxbasis=aux)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("geom"); ap.add_argument("outdir")
    ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--tag", required=True); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--displace", type=float, default=0.0, help="Gaussian displacement per coordinate (A), as dft_timing.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", help="run with GPU4PySCF (mf.to_gpu()); results tagged <setting>_gpu")
    ap.add_argument("--only", default=None, help="comma-separated setting names (default: all)")
    a = ap.parse_args()
    from pyscf import lib
    lib.num_threads(a.threads)
    sym, pos = read_geom(a.geom)
    if a.displace:
        pos = pos + np.random.default_rng(a.seed).normal(0, a.displace, pos.shape)
    os.makedirs(a.outdir, exist_ok=True)
    if a.gpu:          # compile the GPU kernels (RKS or UKS, as the real job) on CH4 / CH3 before anything is timed
        if a.mult == 1:
            run(["C", "H", "H", "H", "H"], np.array([[0, 0, 0], [.63, .63, .63], [-.63, -.63, .63], [-.63, .63, -.63],
                                                     [.63, -.63, -.63]]), 0, 1, "99x590", gpu=True)
        else:
            run(["C", "H", "H", "H"], np.array([[0, 0, 0], [1.08, 0, 0], [-.54, .94, 0], [-.54, -.94, 0]]), 0, 2,
                "99x590", gpu=True)
    for name in (a.only.split(",") if a.only else SETTINGS):
        out = f"{a.outdir}/{a.tag}_{name}{'_gpu' if a.gpu else ''}.json"
        if os.path.exists(out):
            continue
        r = run(sym, pos, a.charge, a.mult, name, gpu=a.gpu)
        r.update(tag=a.tag, geom=os.path.basename(a.geom), displace=a.displace, seed=a.seed, charge=a.charge, mult=a.mult, threads=lib.num_threads(),
                 date=time.strftime("%Y-%m-%d"), host=os.uname().nodename)
        json.dump(r, open(out, "w"))
        print(f"{a.tag} {name:16s} E={r['energy']:.9f} cyc={r['scf_cycles']} t={r['t_total']}s "
              f"ngrid={r['ngrids']} s2={r['s2']:.4f} |mu|={r['dipole_abs']:.4f} fmax={r['fmax']:.2e}", flush=True)


if __name__ == "__main__":
    main()
