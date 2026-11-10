# Rebuild /t/eq_frames.xyz from the 20 T0 species (t0/species_T0.xyz, copied to /t) without PAHdb or container B:
# each species' geometry, then 5 copies with 0.05 A Gaussian displacements, seed 7, the same recipe and order as
# pahdb_select.py. The first 100 frames are the 6,532-atom batch the GH200 timings used (jax_time.py input).
import numpy as np
from ase.io import read, write
frames, prng = [], np.random.default_rng(7)
for i, m in enumerate(read("/t/species_T0.xyz", ":")):
    m.positions -= m.positions.mean(0)
    for k in range(6):
        a = m.copy(); a.positions += (0 if k == 0 else 1) * prng.normal(0, 0.05, a.positions.shape)
        a.info.update(species=i, geom=k); frames.append(a)
write("/t/eq_frames.xyz", frames, format="extxyz")
print(len(frames), "frames;", sum(len(a) for a in frames[:100]), "atoms in the first 100")
