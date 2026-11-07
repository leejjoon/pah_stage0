import torch, numpy as np, time
torch.set_float32_matmul_precision("highest")
import torch_sim as ts, warp
from mace.calculators import mace_off
from mace.calculators.mace_torchsim import MaceTorchSimModel
from ase.build import molecule
print("torch_sim", ts.__version__, "warp", warp.__version__, "torch", torch.__version__)
raw = mace_off(model="small", default_dtype="float32", device="cuda", return_raw_model=True)
mdl = MaceTorchSimModel(model=raw, device=torch.device("cuda"), dtype=torch.float32, compute_forces=True, compute_stress=False)
mols=[molecule("C6H6"), molecule("C6H6"), molecule("CH4")]
F = mdl(ts.io.atoms_to_state(mols, device=torch.device("cuda"), dtype=torch.float32))["forces"].detach().cpu().numpy()
calc = mace_off(model="small", default_dtype="float32", device="cuda"); ref=[]
for a in mols: a.calc=calc; ref.append(a.get_forces())
print("batched vs ASE max|dF| =", np.abs(F-np.concatenate(ref)).max())
try:
    batch=[molecule("C6H6") for _ in range(100)]
    t=time.time()
    out=ts.integrate(system=batch, model=mdl, integrator=ts.Integrator.nvt_langevin, n_steps=200, temperature=600, timestep=0.0005)
    torch.cuda.synchronize(); print(f"100-molecule NVT Langevin, 200 steps: ok in {time.time()-t:.1f}s, n_systems={out.n_systems}")
except Exception as e:
    print("integrate not exercised:", type(e).__name__, str(e)[:200])
