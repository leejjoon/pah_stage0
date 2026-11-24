# Gate G1 force consistency (docs/plan/charged_mlff_plan.md, "Quality checks", item 3): central finite differences of
# the B3LYP/6-31G* energy against the analytic forces of the seed-set labels, with the production settings of
# dft_seed_run.py (GPU4PySCF, container D). For each frame:
#   - reference SCF at the labelled geometry (checked against the label: energy, forces);
#   - analytic gradient as in production (no grid response) and with grid response;
#   - for k force components (default: the 3 largest |F| and 3 random), SCFs at x +/- h (h = 0.005 bohr), each
#     started from the reference density, tighter convergence (1e-11 Ha), <S^2> compared with the reference;
#   - FD = -(E+ - E-) / 2h against both analytic forces.
# The production gradient omits the grid response, so |FD - F_prod| holds that term plus the FD error; |FD - F_grid|
# measures the FD error alone (and so the analytic code). Target (plan): |FD - F_prod| <= 1e-4 Ha/bohr.
#   apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 D_gpu4pyscf.sif python fd_force_check.py \
#       <seed_dft.extxyz> <frame dir with the per-frame JSONs> --names A,B,... --out fd.json [--k-max 3 --k-rand 3]
# Output: one JSON with per-component rows and per-frame maxima; results are appended frame by frame (resumable).
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g16_io import HB_TO_EVA, read_extxyz

BOHR = 0.52917721092


def build(sym, pos, charge, mult):
    from pyscf import gto
    from gpu4pyscf import dft
    mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", cart=True, charge=charge, spin=mult - 1, unit="Angstrom",
                verbose=0)
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol, xc="b3lypg")
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200
    return mf


def scf(mf, dm0=None, tight=False):
    if tight:
        mf.conv_tol = 1e-11; mf.conv_tol_grad = 1e-7
    e = mf.kernel(dm0=dm0); newton = False
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ); newton = True
    s2 = float(mf.spin_square()[0]) if mf.mol.spin else 0.0
    return mf, float(e), newton, s2


def grad(mf, grid_response):
    g = mf.nuc_grad_method(); g.grid_response = grid_response
    g = g.kernel(); return np.asarray(g.get() if hasattr(g, "get") else g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extxyz"); ap.add_argument("jsondir")
    ap.add_argument("--names", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=float, default=0.005, help="displacement in bohr")
    ap.add_argument("--k-max", type=int, default=3); ap.add_argument("--k-rand", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names = a.names.split(",")
    frames = {}
    for info, sym, pos in read_extxyz(a.extxyz):
        n = info.get("log", "").strip('"')
        if n in names:
            frames[n] = (info, sym, np.asarray(pos, float))
    miss = [n for n in names if n not in frames]
    if miss:
        sys.exit(f"not in {a.extxyz}: {miss}")
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for n in names:
        if n in res:
            print(f"{n}: done"); continue
        info, sym, pos = frames[n]
        q = int(info["charge"]); m = int(info["mult"])
        lab = json.load(open(os.path.join(a.jsondir, n + ".json")))
        t0 = time.time()
        mf, e0, nw0, s20 = scf(build(sym, pos, q, m))
        f_prod = -grad(mf, False); f_grid = -grad(mf, True)                       # Ha/bohr
        f_lab = np.asarray(lab["forces_eVA"]) / HB_TO_EVA
        dm0 = mf.make_rdm1()
        flat = np.abs(f_prod).ravel(); order = np.argsort(-flat)
        rng = np.random.default_rng(a.seed + sum(map(ord, n)))
        pick = list(order[:a.k_max])
        rest = [i for i in range(flat.size) if i not in pick]
        pick += list(rng.choice(rest, size=min(a.k_rand, len(rest)), replace=False))
        rows = []
        for idx in pick:
            at, c = divmod(int(idx), 3); es = []; s2s = []; nws = []
            for sgn in (+1, -1):
                p = pos.copy(); p[at, c] += sgn * a.h * BOHR
                mfd = build(sym, p, q, m)
                _, e, nw, s2 = scf(mfd, dm0=dm0, tight=True); es.append(e); s2s.append(s2); nws.append(nw)
            fd = -(es[0] - es[1]) / (2 * a.h)
            rows.append(dict(atom=at, sym=sym[at], comp="xyz"[c], fd=fd, f_prod=float(f_prod[at, c]),
                             f_grid=float(f_grid[at, c]), d_prod=fd - float(f_prod[at, c]),
                             d_grid=fd - float(f_grid[at, c]), ds2=max(abs(s - s20) for s in s2s), newton=any(nws)))
            print(f"  {n} {sym[at]}{at}{'xyz'[c]}: FD {fd:+.7f}  prod {f_prod[at, c]:+.7f}  grid {f_grid[at, c]:+.7f}"
                  f"  dS2 {rows[-1]['ds2']:.1e}", flush=True)
        dp = max(abs(r["d_prod"]) for r in rows); dg = max(abs(r["d_grid"]) for r in rows)
        res[n] = dict(charge=q, mult=m, nC=sym.count("C"), config_type=info.get("config_type", "").strip('"'),
                      e_ref_Ha=e0, e_label_Ha=lab["energy_Ha"], dE_label=e0 - lab["energy_Ha"],
                      dF_label_max=float(np.abs(f_prod - f_lab).max()), grid_term_max=float(np.abs(f_grid - f_prod).max()),
                      s2=s20, newton_ref=nw0, max_abs_F=float(flat.max()), max_d_prod=dp, max_d_grid=dg,
                      max_ds2=max(r["ds2"] for r in rows), h_bohr=a.h, rows=rows, seconds=round(time.time() - t0, 1))
        print(f"{n}: max|FD-F_prod| {dp:.2e}  max|FD-F_grid| {dg:.2e}  grid term (all atoms) "
              f"{res[n]['grid_term_max']:.2e}  label dE {res[n]['dE_label']:.1e} dF {res[n]['dF_label_max']:.1e} Ha/bohr"
              f"  {res[n]['seconds']} s  {'PASS' if dp <= 1e-4 else 'FAIL'}", flush=True)
        json.dump(res, open(a.out + ".tmp", "w"), indent=1); os.replace(a.out + ".tmp", a.out)
    worst = max(r["max_d_prod"] for r in res.values())
    print(f"{len(res)} frames; worst |FD - F_prod| {worst:.2e} Ha/bohr (target 1e-4): {'PASS' if worst <= 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()
