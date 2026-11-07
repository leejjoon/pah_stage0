import os, torch, numpy as np, collections
print("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE =", os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"))
torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32=False
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x=torch.zeros(1).cuda(); print("cuda tensor ok", x.device, "driver-api", torch.cuda.driver_version() if hasattr(torch.cuda,'driver_version') else '')
from ase.io import read
from mace.calculators import mace_off
for cueq in (False, True):
    calc = mace_off(model="small", default_dtype="float32", device="cuda", enable_cueq=cueq)
    mods=collections.Counter(type(m).__module__.split('.')[0] for m in calc.models[0].modules())
    a=read("/t/benzene.xyz"); a.calc=calc; F=a.get_forces(); E=a.get_potential_energy()
    print(f"cueq={cueq} E={E:.6f} cuet_modules={mods.get('cuequivariance_torch',0)}")
    np.save(f"/t/ctr_F_cueq{int(cueq)}.npy", F)
