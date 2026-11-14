# GFN2-xTB molecular dynamics of PAH neutrals and ions, with the same protocol as the neutral MLMD (d1/mlmd_run.py,
# after the authors' 1_MD_calc_position.py / 2_calc_dipole.py), for docs/plan/charged_tdep_validation.md:
# Maxwell-Boltzmann velocities at T from a fixed seed; ASE Langevin NVT (0.5 fs, friction 0.02 per ASE time unit,
# fixcm); then velocity-Verlet NVE; the dipole every 2 steps (1 fs) about the centre of mass (for an ion the dipole
# depends on the origin: mu_com = mu_origin - q R_com).
#   python xtb_md.py run  <xyz> <out.npz> --charge 1 --mult 2 --T 300 --seed 0 [--nvt-ps 20] [--nve-ps 100]
#   python xtb_md.py time <xyz> --charge 1 --mult 2 [--steps 200]          # seconds per MD step
# One trajectory per process; set OMP_NUM_THREADS (1-4) and run many in parallel. The output npz holds dipole
# (frames, 3) in e*A every 1 fs, T and E_tot every 100 frames, and is rewritten every 10,000 frames (partial runs
# keep what they have; frames_done says how far). Needs ase and tblite (conda-forge tblite-python).
import argparse, time
import numpy as np


def calculator(charge, mult):
    from tblite.ase import TBLite
    return TBLite(method="GFN2-xTB", charge=charge, multiplicity=mult, verbosity=0)


def com(atoms):
    m = atoms.get_masses()
    return (m[:, None] * atoms.positions).sum(0) / m.sum()


def dipole_com(atoms, charge):
    return atoms.get_dipole_moment() - charge * com(atoms)


def load(a):
    from ase.io import read
    atoms = read(a.xyz)
    atoms.calc = calculator(a.charge, a.mult)
    return atoms


def cmd_time(a):
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = load(a)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))
    md = VelocityVerlet(atoms, 0.5 * units.fs)
    md.run(5)
    t0 = time.time(); md.run(a.steps); dt = (time.time() - t0) / a.steps
    print(f"{a.xyz}: {len(atoms)} atoms, charge {a.charge}, mult {a.mult}: {dt * 1e3:.1f} ms per step; "
          f"{dt * 2000:.1f} s per ps; dipole {dipole_com(atoms, a.charge)}")


def cmd_run(a):
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    atoms = load(a)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.T, rng=np.random.default_rng(a.seed))
    dt = 0.5 * units.fs
    n_nvt, n_nve = int(round(a.nvt_ps * 2000)), int(round(a.nve_ps * 2000))
    t0 = time.time()
    nvt = Langevin(atoms, dt, temperature_K=a.T, friction=0.02, fixcm=True, rng=np.random.default_rng(a.seed + 1))
    nvt.run(n_nvt)
    n_frames = n_nve // 2
    mu = np.full((n_frames, 3), np.nan, np.float32)
    qa = []
    md = VelocityVerlet(atoms, dt)

    def save(done):
        np.savez(a.out, dipole=mu[:done], qa=np.array(qa), frames_done=done, xyz=a.xyz, charge=a.charge,
                 mult=a.mult, T=a.T, seed=a.seed, nvt_ps=a.nvt_ps, nve_ps=a.nve_ps, dt_fs=1.0,
                 seconds=time.time() - t0, positions=atoms.positions)

    for k in range(n_frames):
        md.run(2)
        mu[k] = dipole_com(atoms, a.charge)
        if k % 100 == 0:
            qa.append((k, atoms.get_temperature(), atoms.get_potential_energy() + atoms.get_kinetic_energy()))
        if (k + 1) % 10000 == 0:
            save(k + 1)
    save(n_frames)
    print(f"{a.out}: {n_frames} frames in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "time"):
        p = sub.add_parser(name); p.add_argument("xyz")
        if name == "run":
            p.add_argument("out"); p.add_argument("--T", type=float, required=True); p.add_argument("--seed", type=int, default=0)
            p.add_argument("--nvt-ps", type=float, default=20.0); p.add_argument("--nve-ps", type=float, default=100.0)
        else:
            p.add_argument("--steps", type=int, default=200)
        p.add_argument("--charge", type=int, default=0); p.add_argument("--mult", type=int, default=1)
    a = ap.parse_args()
    (cmd_run if a.cmd == "run" else cmd_time)(a)


if __name__ == "__main__":
    main()
