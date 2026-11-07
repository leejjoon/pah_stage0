import numpy as np
J={k:np.load(f"/t/{f}")["forces"][0] for k,f in (("jax","ctr_jax.npz"),("jax_cueq","ctr_jax_cueq.npz"))}
for k,Fj in J.items():
    for c in (0,1):
        print(f"{k:9s} vs torch cueq={c}: max|dF| = {np.abs(Fj-np.load(f'/t/ctr_F_cueq{c}.npy')).max():.2e} eV/A")
