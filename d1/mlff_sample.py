# Training geometries for charged-state MLFFs (docs/plan/charged_mlff_plan.md): GFN2-xTB Langevin MD of one species
# in one charge state at several temperatures (cheap, handles ions), frames taken after equilibration and spaced to
# decorrelate, plus Gaussian displacements of the input (DFT-optimised) geometry. Frames whose bond graph differs
# from the input (bond breaking, H hops) are dropped and counted: the seed set is for intact molecules.
#   python mlff_sample.py <xyz> <out.extxyz> --charge 1 --mult 2 [--temps 300,600,900,1200,1500] [--per-T 4]
#        [--equil-ps 2] [--gap-ps 0.5] [--n-disp 5] [--disp 0.05] [--seed 0]
# Each frame is centred on its centre of mass (the dipole of an ion depends on the origin). Output: extended XYZ with
# charge, mult, config_type ("md_<T>K" or "disp"), sample_T and the source file in the comment line.
# Needs ase and tblite (the micromamba env /var/tmp/$USER-pah/mamba/xtb). One core per process.
import argparse, os
import numpy as np

RCOV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}


def bonds(sym, p):
    r = np.array([RCOV[s] for s in sym]); d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    return frozenset(zip(*np.nonzero(np.triu((d < 1.2 * (r[:, None] + r[None])) & ~np.eye(len(sym), dtype=bool)))))


def centred(atoms):
    m = atoms.get_masses(); p = atoms.positions
    return p - (m[:, None] * p).sum(0) / m.sum()


def write_frame(f, sym, p, info):
    f.write(f"{len(sym)}\n" + 'Properties=species:S:1:pos:R:3 ' + " ".join(f"{k}={v}" for k, v in info.items()) + "\n")
    f.writelines(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(sym, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz"); ap.add_argument("out")
    ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--temps", default="300,600,900,1200,1500"); ap.add_argument("--per-T", type=int, default=4)
    ap.add_argument("--equil-ps", type=float, default=2.0); ap.add_argument("--gap-ps", type=float, default=0.5)
    ap.add_argument("--n-disp", type=int, default=5); ap.add_argument("--disp", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from ase import units
    from ase.constraints import FixCom
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from tblite.ase import TBLite
    ref = read(a.xyz); sym = ref.get_chemical_symbols(); b0 = bonds(sym, ref.positions)
    src = os.path.basename(a.xyz); rng = np.random.default_rng(a.seed)
    kept = dropped = 0
    tmp = a.out + ".part"
    with open(tmp, "w") as f:
        base = dict(charge=a.charge, mult=a.mult, source=src)
        for k in range(a.n_disp):
            p = ref.positions + rng.normal(0, a.disp, ref.positions.shape)
            at = ref.copy(); at.positions = p
            write_frame(f, sym, centred(at), {**base, "config_type": "disp", "sample_T": 0}); kept += 1
        for T in [float(t) for t in a.temps.split(",")]:
            at = ref.copy(); at.set_constraint(FixCom())
            at.calc = TBLite(method="GFN2-xTB", charge=a.charge, multiplicity=a.mult, verbosity=0)
            MaxwellBoltzmannDistribution(at, temperature_K=T, rng=np.random.default_rng(rng.integers(2**31)))
            md = Langevin(at, 0.5 * units.fs, temperature_K=T, friction=0.02, fixcm=False,
                          rng=np.random.default_rng(rng.integers(2**31)))
            md.run(int(round(a.equil_ps * 2000)))
            for _ in range(a.per_T):
                md.run(int(round(a.gap_ps * 2000)))
                if bonds(sym, at.positions) != b0:
                    dropped += 1; continue
                write_frame(f, sym, centred(at), {**base, "config_type": f"md_{T:.0f}K", "sample_T": f"{T:.0f}"})
                kept += 1
    os.replace(tmp, a.out)
    print(f"{a.out}: {kept} frames, {dropped} dropped (bond graph changed)")


if __name__ == "__main__":
    main()
