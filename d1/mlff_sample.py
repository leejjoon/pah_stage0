# Training geometries for charged-state MLFFs (docs/plan/charged_mlff_plan.md), one species in one charge state.
# Every candidate frame is kept and labelled; nothing is dropped (committee review B2: dropping frames whose bond
# graph changed censors the target distribution). Frame families (config_type):
#   min         the input (DFT-optimised) geometry, undisplaced
#   disp        Gaussian random displacements of the minimum (--n-disp, --disp A per coordinate)
#   nms_<T>K    normal-mode sampling: harmonic thermal displacements at T along the normal modes of a GFN2-xTB
#               finite-difference Hessian at the minimum (--nms "300:2,1000:2")
#   md_<T>K     GFN2-xTB Langevin MD (centre of mass fixed): --equil-ps, then --per-T frames --gap-ps apart
#   bound_ch    controlled C-H stretches of one random H to --ch-lengths (A), at the edge of and past the bond cut
#   bound_oop   one random atom pushed out of the molecular plane by --oop-range (A)
# Labels on every frame: sample_T (MD/NMS temperature, "none" otherwise); max_stretch (largest d/d0 over the input's
# bonds) and max_elong (largest d - d0, A); graph_changed (bond graph at 1.2 x covalent radii differs from the input).
# MD frames also carry changed_n: of three checks (the frame, +50 fs, +100 fs), how many show a changed graph
# (3 = persistent, 1-2 = transient), and after_change: whether this trajectory had a persistent change earlier.
#   python mlff_sample.py <xyz> <out.extxyz> --charge 1 --mult 2 [--temps 300,600,900,1200,1500] [--per-T 4]
#        [--equil-ps 2] [--gap-ps 0.5] [--n-disp 3] [--disp 0.05] [--nms 300:2,1000:2] [--n-ch 2]
#        [--ch-lengths 1.25,1.40] [--n-oop 2] [--oop-range 0.3,0.6] [--seed 0]
# Frames are centred on the centre of mass (an ion's dipole depends on the origin). Needs ase and tblite (micromamba
# env /var/tmp/$USER-pah/mamba/xtb). One core per process.
import argparse, os
import numpy as np

RCOV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
KB_EV = 8.617333262e-5


def bond_list(sym, p):
    r = np.array([RCOV[s] for s in sym]); d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    i, j = np.nonzero(np.triu((d < 1.2 * (r[:, None] + r[None])) & ~np.eye(len(sym), dtype=bool)))
    return i, j, d[i, j]


def graph(sym, p):
    i, j, _ = bond_list(sym, p)
    return frozenset(zip(i.tolist(), j.tolist()))


def labels(sym, p, ref):
    """Stretch of the reference bonds and whether the bond graph changed."""
    i, j, d0 = ref["bonds"]
    d = np.linalg.norm(p[i] - p[j], axis=1)
    return {"max_stretch": f"{(d / d0).max():.4f}", "max_elong": f"{(d - d0).max():.4f}",
            "graph_changed": int(graph(sym, p) != ref["graph"])}


def centred(p, m):
    return p - (m[:, None] * p).sum(0) / m.sum()


def write_frame(f, sym, p, info):
    f.write(f"{len(sym)}\n" + 'Properties=species:S:1:pos:R:3 ' + " ".join(f"{k}={v}" for k, v in info.items()) + "\n")
    f.writelines(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(sym, p))


def hessian(atoms, calc, h=0.005):
    """Central finite-difference Hessian (eV/A^2) from xTB forces."""
    a = atoms.copy(); a.calc = calc; p0 = a.positions.copy(); n = len(a)
    H = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        for s in (1, -1):
            p = p0.copy(); p[k // 3, k % 3] += s * h; a.positions = p
            H[k] -= s * a.get_forces().ravel() / (2 * h)
    return 0.5 * (H + H.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz"); ap.add_argument("out")
    ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--temps", default="300,600,900,1200,1500"); ap.add_argument("--per-T", type=int, default=4)
    ap.add_argument("--equil-ps", type=float, default=2.0); ap.add_argument("--gap-ps", type=float, default=0.5)
    ap.add_argument("--n-disp", type=int, default=3); ap.add_argument("--disp", type=float, default=0.05)
    ap.add_argument("--nms", default="300:2,1000:2", help="T:count pairs for normal-mode sampling ('' = none)")
    ap.add_argument("--n-ch", type=int, default=2); ap.add_argument("--ch-lengths", default="1.25,1.40")
    ap.add_argument("--n-oop", type=int, default=2); ap.add_argument("--oop-range", default="0.3,0.6")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from ase import units
    from ase.constraints import FixCom
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from tblite.ase import TBLite
    xtb = lambda: TBLite(method="GFN2-xTB", charge=a.charge, multiplicity=a.mult, verbosity=0)
    ref_atoms = read(a.xyz); sym = ref_atoms.get_chemical_symbols(); p_ref = ref_atoms.positions.copy()
    mass = ref_atoms.get_masses()
    ref = {"bonds": bond_list(sym, p_ref), "graph": graph(sym, p_ref)}
    rng = np.random.default_rng(a.seed)
    base = dict(charge=a.charge, mult=a.mult, source=os.path.basename(a.xyz))
    counts = {}
    tmp = a.out + ".part"
    with open(tmp, "w") as f:
        def emit(p, ctype, T="none", **extra):
            write_frame(f, sym, centred(p, mass), {**base, "config_type": ctype, "sample_T": T, **labels(sym, p, ref), **extra})
            counts[ctype.split("_")[0]] = counts.get(ctype.split("_")[0], 0) + 1

        emit(p_ref, "min")
        for _ in range(a.n_disp):
            emit(p_ref + rng.normal(0, a.disp, p_ref.shape), "disp")

        pairs = [tuple(x.split(":")) for x in a.nms.split(",") if x]
        if pairs:                                        # harmonic thermal sampling along xTB normal modes
            H = hessian(ref_atoms, xtb())
            w = 1 / np.sqrt(np.repeat(mass, 3))          # mass-weighting (amu)
            ev, vec = np.linalg.eigh(H * w[:, None] * w[None])
            ok = ev > 1e-3                               # drop negative and near-zero modes ...
            ok[np.argsort(np.abs(ev))[:6]] = False       # ... and the six translations/rotations (the PAHdb minimum is
                                                         # not an xTB minimum, so they need not come out at zero)
            for T, n in pairs:
                kT = KB_EV * float(T)
                for _ in range(int(n)):
                    q = rng.normal(0, np.sqrt(kT / ev[ok]))   # <q^2> = kT / omega^2 in mass-weighted units
                    emit(p_ref + (w * (vec[:, ok] @ q)).reshape(-1, 3), f"nms_{float(T):.0f}K", T=f"{float(T):.0f}")

        hs = [k for k, s in enumerate(sym) if s == "H"]
        for L in [float(x) for x in a.ch_lengths.split(",")][:a.n_ch] if hs else []:
            h = int(rng.choice(hs)); i, j, _ = ref["bonds"]
            nb = [int(j[k]) if i[k] == h else int(i[k]) for k in range(len(i)) if h in (i[k], j[k]) and sym[int(j[k]) if i[k] == h else int(i[k])] == "C"]
            if not nb:
                continue
            c = nb[0]; u = p_ref[h] - p_ref[c]; u /= np.linalg.norm(u)
            p = p_ref.copy(); p[h] = p_ref[c] + L * u
            emit(p, "bound_ch", bound_atom=h, bound_value=f"{L:.3f}")
        lo, hi = [float(x) for x in a.oop_range.split(",")]
        c0 = p_ref - p_ref.mean(0); normal = np.linalg.svd(c0)[2][-1]   # least-variance axis = plane normal
        for _ in range(a.n_oop):
            k = int(rng.integers(len(sym))); dz = rng.uniform(lo, hi) * rng.choice([-1, 1])
            p = p_ref.copy(); p[k] += dz * normal
            emit(p, "bound_oop", bound_atom=k, bound_value=f"{dz:.3f}")

        n_check = 100                                    # 50 fs between persistence checks
        for T in [float(t) for t in a.temps.split(",")]:
            at = ref_atoms.copy(); at.set_constraint(FixCom()); at.calc = xtb()
            MaxwellBoltzmannDistribution(at, temperature_K=T, rng=np.random.default_rng(rng.integers(2**31)))
            md = Langevin(at, 0.5 * units.fs, temperature_K=T, friction=0.02, fixcm=False,
                          rng=np.random.default_rng(rng.integers(2**31)))
            md.run(int(round(a.equil_ps * 2000)))
            changed_before = 0
            for _ in range(a.per_T):
                md.run(int(round(a.gap_ps * 2000)) - 2 * n_check)
                p = at.positions.copy(); n_changed = int(graph(sym, p) != ref["graph"])
                for _ in range(2):
                    md.run(n_check); n_changed += int(graph(sym, at.positions) != ref["graph"])
                emit(p, f"md_{T:.0f}K", T=f"{T:.0f}", changed_n=n_changed, after_change=changed_before)
                changed_before = max(changed_before, int(n_changed == 3))
    os.replace(tmp, a.out)
    print(f"{a.out}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
