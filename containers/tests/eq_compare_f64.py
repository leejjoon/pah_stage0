# Energy part of plan 4.2 in float64 on both stacks (option c): jax --dtype float64 vs torch default_dtype float64.
import numpy as np
from ase.io import read
nat = np.array([len(a) for a in read("/t/eq_frames.xyz", ":")])
E64t = np.load("/t/eq_torch_f64_energy.npy"); J = np.load("/t/eq_jax_f64.npz", allow_pickle=True)
dE = np.abs(J["energy"].astype(float) - E64t) / nat
print(f"jax float64 dtype {J['energy'].dtype}; vs torch float64: max|dE|/atom {dE.max():.1e} eV, max|dE| {np.abs(J['energy'] - E64t).max():.1e} eV  -> {'PASS' if dE.max() < 1e-4 else 'FAIL'} (gate 1e-4 eV/atom)")
