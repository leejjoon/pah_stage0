# Born-Oppenheimer DFT MD reference for the charged-state MLFF work (committee review B3, part one; docs/plan/
# charged_mlff_plan.md): B3LYP (Gaussian's VWN3, "b3lypg")/6-31G* with PySCF on CPU, the level of the training data.
# It measures the ML approximation error at that level of theory along real DFT dynamics (forces, dipoles, spectra
# and their temperature trends); it does not test classical nuclear dynamics itself.
# Protocol as the MLMD and xTB runs: Maxwell-Boltzmann start at T, Langevin NVT (friction 0.02 per ASE time unit,
# centre of mass fixed), then velocity-Verlet NVE, 0.5 fs; dipole about the centre of mass every 2 steps (1 fs);
# total energy, temperature and <S^2> every 20 steps. Each SCF starts from the previous step's density matrix.
# Basis functions: Cartesian (6D) d functions by default, as Gaussian uses for 6-31G* (and so the training data and
# PAHdb v4); PySCF's own default is spherical (5D). --spherical switches back.
#   python aimd_pyscf.py <xyz> <out.npz> --charge 1 --mult 2 --T 900 --seed 0 [--nvt-ps 2] [--nve-ps 20] [--threads 8]
#   python aimd_pyscf.py <xyz> - --charge 1 --mult 2 --time-steps 10        (seconds per step, then exit)
# The npz is rewritten every --ckpt-fs (default 100 fs) with positions, velocities and the step; rerunning the same
# command resumes from it (the NVT thermostat restarts with a fresh random stream). Needs the micromamba env
# /var/tmp/$USER-pah/mamba/dft (pyscf) and ase.
import argparse, os, time
import numpy as np


class PySCFCalc:
    """Minimal ASE calculator: energy, forces and centre-of-mass dipole from PySCF RKS/UKS, density reused."""
    implemented_properties = ["energy", "forces"]

    def __init__(self, charge, mult, basis="6-31g*", xc="b3lypg", conv_tol=1e-9, cart=True):
        self.q, self.m, self.basis, self.xc, self.tol, self.cart = charge, mult, basis, xc, conv_tol, cart
        self.dm = None; self.cycles = 0; self.last = None

    def compute(self, atoms):
        from pyscf import gto, dft
        from ase.units import Hartree, Bohr
        pos = atoms.positions
        if self.last is not None and np.array_equal(pos, self.last[0]):
            return self.last[1]
        mol = gto.M(atom=list(zip(atoms.get_chemical_symbols(), pos)), basis=self.basis, charge=self.q,
                    spin=self.m - 1, unit="Angstrom", cart=self.cart, verbose=0)
        mf = (dft.RKS if self.m == 1 else dft.UKS)(mol); mf.xc = self.xc; mf.conv_tol = self.tol; mf.max_cycle = 200
        n = [0]; mf.callback = lambda env: n.__setitem__(0, n[0] + 1)
        e = mf.kernel(dm0=self.dm)
        if not mf.converged:
            mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ)   # newton().kernel takes MOs first, not a density
        self.dm = mf.make_rdm1(); self.cycles = n[0]
        g = mf.nuc_grad_method().kernel()
        mass = atoms.get_masses(); com = (mass[:, None] * pos).sum(0) / mass.sum()
        mu = np.asarray(mf.dip_moment(unit="Debye", verbose=0)) / 4.80320471 - self.q * com     # e*A about the COM
        s2 = float(mf.spin_square()[0]) if self.m > 1 else 0.0
        res = dict(energy=e * Hartree, forces=-g * Hartree / Bohr, dipole=mu, s2=s2)
        self.last = (pos.copy(), res)
        return res


def make_atoms(xyz, calc):
    from ase.io import read
    from ase.calculators.calculator import Calculator, all_changes

    class Wrap(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            r = calc.compute(self.atoms); self.results = {"energy": r["energy"], "forces": r["forces"]}

    at = read(xyz); at.calc = Wrap()
    return at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz"); ap.add_argument("out")
    ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--T", type=float, default=300.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nvt-ps", type=float, default=2.0); ap.add_argument("--nve-ps", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--ckpt-fs", type=float, default=100.0)
    ap.add_argument("--time-steps", type=int, default=0, help="time this many MD steps and exit")
    ap.add_argument("--spherical", action="store_true", help="5D d functions (PySCF default) instead of Gaussian's 6D")
    a = ap.parse_args()
    from pyscf import lib
    from ase import units
    from ase.constraints import FixCom
    from ase.md.langevin import Langevin
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    lib.num_threads(a.threads)
    calc = PySCFCalc(a.charge, a.mult, cart=not a.spherical); at = make_atoms(a.xyz, calc); at.set_constraint(FixCom())
    dt = 0.5 * units.fs
    n_nvt, n_nve = int(round(a.nvt_ps * 2000)), int(round(a.nve_ps * 2000)); n_tot = n_nvt + n_nve
    ck = int(round(a.ckpt_fs * 2))
    if a.time_steps:
        MaxwellBoltzmannDistribution(at, temperature_K=a.T, rng=np.random.default_rng(a.seed))
        md = VelocityVerlet(at, dt); md.run(1)
        t0 = time.time(); md.run(a.time_steps); s = (time.time() - t0) / a.time_steps
        print(f"{a.xyz} q={a.charge}: {s:.2f} s per step at {a.threads} threads, SCF cycles at the last step "
              f"{calc.cycles}; {s * 2000 / 3600:.2f} h per ps; {s * n_tot / 3600:.0f} h for {n_tot} steps")
        return
    step = 0; mu = np.full((n_nve // 2, 3), np.nan); qa = []
    if os.path.exists(a.out):
        z = np.load(a.out); step = int(z["step"]); at.positions = z["positions"]; at.set_velocities(z["velocities"])
        mu[:len(z["dipole"])] = z["dipole"]; qa = [tuple(r) for r in z["qa"]]
        print(f"resuming {a.out} at step {step} of {n_tot}")
    else:
        MaxwellBoltzmannDistribution(at, temperature_K=a.T, rng=np.random.default_rng(a.seed))
    t0 = time.time()

    def save():
        k = max(0, (step - n_nvt) // 2)
        np.savez(a.out + ".tmp.npz", step=step, positions=at.positions, velocities=at.get_velocities(), dipole=mu[:k],
                 qa=np.array(qa), xyz=a.xyz, charge=a.charge, mult=a.mult, T=a.T, seed=a.seed, nvt_ps=a.nvt_ps,
                 nve_ps=a.nve_ps, dt_fs=1.0, method="B3LYP(VWN3)/6-31G* PySCF " + ("5D" if not calc.cart else "6D"), seconds=time.time() - t0)
        os.replace(a.out + ".tmp.npz", a.out)

    while step < n_tot:
        nvt = step < n_nvt
        stop = min(n_nvt if nvt else n_tot, step + ck)
        md = (Langevin(at, dt, temperature_K=a.T, friction=0.02, fixcm=False,
                       rng=np.random.default_rng(a.seed * 1000 + step)) if nvt else VelocityVerlet(at, dt))
        while step < stop:
            md.run(1); step += 1
            if not nvt and (step - n_nvt) % 2 == 0:
                mu[(step - n_nvt) // 2 - 1] = calc.last[1]["dipole"]
            if step % 20 == 0:
                qa.append((step, at.get_temperature(), at.get_potential_energy() + at.get_kinetic_energy(),
                           calc.last[1]["s2"], calc.cycles))
        save()
    print(f"{a.out}: done, {n_tot} steps in {time.time() - t0:.0f} s this session")


if __name__ == "__main__":
    main()
