# Plan 4.2 (revised 2026-11-08), PyTorch side, on the 120 frames in /t/eq_frames.xyz.
# For float32 and float64, cuEquivariance off and on: total energy and forces from the ASE calculator, and the
# interaction energy taken directly from the model (out["interaction_energy"], before E0 is added) on the
# calculator's own batch, dtype and backend. Prints the effective math dtype of every cuEquivariance module.
# Usage: eq_torch_gates.py [n_frames]   (n_frames for a quick probe; default all)
import sys, numpy as np, torch
torch.set_float32_matmul_precision("highest"); torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
from ase.io import read
from mace.calculators import mace_off
from mace.tools import torch_tools

fr = read("/t/eq_frames.xyz", ":")
if len(sys.argv) > 1: fr = fr[: int(sys.argv[1])]
print("TF32 cuBLAS override env:", __import__("os").environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
      "| matmul allow_tf32:", torch.backends.cuda.matmul.allow_tf32, "| cudnn allow_tf32:", torch.backends.cudnn.allow_tf32)


def cueq_dtypes(model):
    """Effective dtypes of cuEquivariance modules: (class, attribute, value), deduplicated."""
    seen = set()
    for name, mod in model.named_modules():
        if not type(mod).__module__.startswith("cuequivariance"): continue
        for attr in ("math_dtype", "dtype"):
            v = getattr(mod, attr, None)
            if v is not None: seen.add((type(mod).__name__, attr, str(v)))
        for sub in vars(mod).values():          # e.g. a SegmentedPolynomial held as a plain attribute
            v = getattr(sub, "math_dtype", None)
            if v is not None: seen.add((type(mod).__name__ + "." + type(sub).__name__, "math_dtype", str(v)))
        for bname, b in mod.named_buffers(recurse=False):
            if b.is_floating_point(): seen.add((type(mod).__name__, "buffer:" + bname, str(b.dtype)))
        for pname, p in mod.named_parameters(recurse=False):
            seen.add((type(mod).__name__, "param:" + pname, str(p.dtype)))
    return sorted(seen)


def interaction_energy(calc, atoms):
    model = calc.models[0]
    assert type(model).__name__ == "ScaleShiftMACE", type(model).__name__
    batch = calc._atoms_to_batch(atoms)
    dt = next(model.parameters()).dtype
    for k in batch.keys:
        v = batch[k]
        if torch.is_tensor(v) and torch.is_floating_point(v): batch[k] = v.to(dtype=dt)
    with torch_tools.default_dtype(calc.default_dtype):
        out = model(batch.to_dict(), compute_force=False, training=False)
    return out["interaction_energy"].detach().cpu().double().item(), str(out["interaction_energy"].dtype)


for dtype in ("float32", "float64"):
    for cueq in (False, True):
        calc = mace_off(model="small", default_dtype=dtype, device="cuda", enable_cueq=cueq)
        model = calc.models[0]
        print(f"== torch {dtype} cueq={int(cueq)}: model {type(model).__name__}, param dtypes {sorted({str(p.dtype) for p in model.parameters()})}")
        if cueq:
            d = cueq_dtypes(model)
            for row in d: print("   cueq module", *row)
            if not d: print("   no cuequivariance modules found")
        E, Ei, Ei_node, F = [], [], [], []
        for a in fr:
            a.calc = calc
            E.append(a.get_potential_energy()); F.append(a.get_forces())
            Ei_node.append(np.asarray(calc.results["node_energy"], np.float64).sum())   # cross-check only
            e, edt = interaction_energy(calc, a); Ei.append(e)
        E, Ei, Ei_node = map(np.array, (E, Ei, Ei_node))
        print(f"   interaction_energy dtype {edt}; |E_int(model) - sum node_energy| max {np.abs(Ei - Ei_node).max():.1e} eV")
        np.savez(f"/t/torch_{dtype}_cueq{int(cueq)}.npz", energy=E, interaction_energy=Ei, interaction_energy_node=Ei_node,
                 forces=np.array(F, dtype=object), allow_pickle=True)
print("torch done:", len(fr), "frames")
