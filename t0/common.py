"""Shared, numpy-only pieces of the T0 MD drivers (plan 4.3), importable from containers A and B.

- the 20 species (DFT geometries) in t0/species_T0.xyz, in the order of t0/species_T0.csv
- the batch: b trajectories cycling through the 20 species (trajectory k is species k % 20)
- identical initial velocities for every stack: Maxwell-Boltzmann at 600 K from one seed per trajectory,
  with no centre-of-mass removal (COM policy: unconstrained throughout; temperature uses 3N degrees of freedom)
- the dipole proxy: fixed charges, -0.1 e on each C bonded to H and +0.1 e on each H, times positions
Units here: Angstrom, amu, eV, fs. Velocities are in Angstrom/fs.
"""
import numpy as np

KB = 8.617333262e-5          # eV/K
T_K = 600.0
DT_FS = 0.5
GAMMA_PER_FS = 0.01
FS_PER_TU = 10.1805054836    # one time unit in eV/Angstrom/amu, in fs (ASE: units.fs = 1/10.1805...)
AMU_A2_PER_FS2_IN_EV = 103.6426965   # 1 amu A^2/fs^2 = 103.64 eV
MASS = {1: 1.008, 6: 12.011}  # amu, same values as ASE's atomic_masses for H and C to 3 decimals; both drivers use these
EQUIL_STEPS, PROD_STEPS = 2000, 20000


def read_xyz(path):
    """Minimal extended-XYZ reader (Z, positions) so that the JAX container needs no ASE."""
    sym = {"H": 1, "C": 6}
    mols, lines = [], open(path).read().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip(): i += 1; continue
        n = int(lines[i]); body = lines[i + 2:i + 2 + n]
        Z = np.array([sym[l.split()[0]] for l in body]); X = np.array([[float(v) for v in l.split()[1:4]] for l in body])
        mols.append((Z, X)); i += 2 + n
    return mols


def batch(species, b):
    """Trajectory k uses species k % len(species)."""
    return [species[k % len(species)] for k in range(b)]


def initial_velocities(Z, k, seed=20261108):
    """Maxwell-Boltzmann velocities (Angstrom/fs) at 600 K for trajectory k; no COM removal."""
    m = np.array([MASS[z] for z in Z])
    rng = np.random.default_rng([seed, k])
    sigma = np.sqrt(KB * T_K / (m * AMU_A2_PER_FS2_IN_EV))     # Angstrom/fs
    return rng.normal(size=(len(Z), 3)) * sigma[:, None]


def proxy_charges(Z, X):
    """-0.1 e on each C bonded to an H, +0.1 e on each H (fixed from the initial geometry)."""
    q = np.zeros(len(Z))
    H = np.where(Z == 1)[0]; C = np.where(Z == 6)[0]
    for h in H:
        c = C[np.argmin(np.linalg.norm(X[C] - X[h], axis=1))]
        q[h] += 0.1; q[c] -= 0.1
    return q


def temperature(v, m):
    """Kinetic temperature (K) with 3N degrees of freedom; v in Angstrom/fs, m in amu."""
    ke = 0.5 * (m[:, None] * v ** 2).sum() * AMU_A2_PER_FS2_IN_EV
    return 2 * ke / (3 * len(m) * KB)


def vdos(v, dt_fs):
    """Vibrational density of states from velocities v[t, atom, 3] (mass-unweighted); returns (cm^-1, power)."""
    v = v - v.mean(0)
    f = np.fft.rfft(v, axis=0, n=2 * len(v))
    p = (np.abs(f) ** 2).sum((1, 2))
    nu = np.fft.rfftfreq(2 * len(v), d=dt_fs * 1e-15) / 2.99792458e10
    return nu, p
