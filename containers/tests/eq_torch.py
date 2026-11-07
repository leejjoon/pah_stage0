import numpy as np, torch
torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = False
from ase.io import read
from mace.calculators import mace_off
fr = read("/t/eq_frames.xyz", ":")
for cueq in (False, True):
    calc = mace_off(model="small", default_dtype="float32", device="cuda", enable_cueq=cueq)
    E, F = [], []
    for a in fr: a.calc = calc; E.append(a.get_potential_energy()); F.append(a.get_forces())
    np.savez(f"/t/eq_torch_cueq{int(cueq)}.npz", energy=np.array(E), forces=np.array(F, dtype=object), allow_pickle=True)
    print(f"torch cueq={cueq}: {len(fr)} frames")
calc = mace_off(model="small", default_dtype="float64", device="cuda")   # reference for the float32 rounding floor
E = []
for a in fr: a.calc = calc; E.append(a.get_potential_energy())
np.save("/t/eq_torch_f64_energy.npy", np.array(E)); print("torch float64 reference done")
