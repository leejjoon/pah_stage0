# T2.5 of docs/plan/handoff_pyscf_gpu_jax.md: throughput of GPU4PySCF single points (energy, forces, dipole about the
# centre of mass, <S^2>) when one process works through many frames of one species, and when several such processes
# share one GPU. Settings are the ones matched to Gaussian in T1 (b3lypg, 6-31G* with 6D d, (99, 590) grid,
# conv_tol 1e-9 with conv_tol_grad 1e-6). Frames are the input geometry with Gaussian displacements (--displace A, seeds seed0..seed0+n-1),
# like the seed set's distorted frames.
#   python dft_gpu_batch.py <xyz> --charge 1 --mult 2 --n 8 --seed0 0 [--reuse-dm] [--out run.json]
# Prints one line per frame and a summary (wall time after warm-up, points per hour, GPU memory).
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyscf_match import read_geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz"); ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--displace", type=float, default=0.05)
    ap.add_argument("--reuse-dm", action="store_true", help="start each SCF from the previous frame's density")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import cupy
    from pyscf import gto
    from gpu4pyscf import dft
    sym, pos0 = read_geom(a.xyz)

    def point(pos, dm0=None):
        mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", charge=a.charge, spin=a.mult - 1, unit="Angstrom",
                    cart=True, verbose=0)
        mf = (dft.RKS if a.mult == 1 else dft.UKS)(mol, xc="b3lypg"); mf.grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200   # GPU noise floor ~1e-9 Ha
        e = mf.kernel(dm0=dm0)
        if not mf.converged:
            mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ)   # newton().kernel takes MOs first, not a density
        g = mf.nuc_grad_method().kernel()
        mu = mf.dip_moment(unit="Debye", verbose=0)
        s2 = mf.spin_square()[0] if a.mult > 1 else 0.0
        return float(e), bool(mf.converged), int(getattr(mf, "cycles", 0)), (mf.make_rdm1() if a.reuse_dm else None)

    point(pos0)                                                # warm-up: kernel compilation, not timed
    t_all = time.time(); rows = []; dm = None
    for k in range(a.n):
        pos = pos0 + np.random.default_rng(a.seed0 + k).normal(0, a.displace, pos0.shape)
        t = time.time(); e, ok, cyc, dm = point(pos, dm)
        rows.append(dict(seed=a.seed0 + k, energy=e, converged=ok, cycles=cyc, t=round(time.time() - t, 2)))
        print(f"{os.path.basename(a.xyz)} seed {a.seed0 + k} E={e:.8f} conv={ok} cyc={cyc} t={rows[-1]['t']}s", flush=True)
    wall = time.time() - t_all
    r = dict(xyz=os.path.basename(a.xyz), charge=a.charge, mult=a.mult, n=a.n, reuse_dm=a.reuse_dm, wall=round(wall, 1),
             s_per_point=round(wall / a.n, 2), points_per_hour=round(3600 * a.n / wall, 1),
             gpu_mem_gb=round(cupy.get_default_memory_pool().total_bytes() / 1e9, 2), frames=rows,
             date=time.strftime("%Y-%m-%d"), host=os.uname().nodename,
             gpu=cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print(json.dumps({k: v for k, v in r.items() if k != "frames"}), flush=True)
    if a.out:
        json.dump(r, open(a.out, "w"))


if __name__ == "__main__":
    main()
