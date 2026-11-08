# Verify the x64 mace-jax bundle against the float64 torch checkpoint (plan 4.2):
# every floating torch parameter/buffer must appear in the bundle at float64 precision.
# The match is by value set per tensor, so it is independent of the converter's reshapes/transposes.
import sys, numpy as np, torch
from flax import serialization
from mace.calculators import mace_off
bundle = sys.argv[1] if len(sys.argv) > 1 else "/t/off_small_x64.npz"
def leaves(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items(): yield from leaves(v, p + "/" + str(k))
    else: yield p, np.asarray(d)
J = dict(leaves(serialization.msgpack_restore(open(bundle, "rb").read())))
for p, a in J.items():
    if a.dtype.kind == "f" and a.dtype != np.float64: print(f"non-float64 leaf: {p} {a.dtype} shape {a.shape} values {a.ravel()[:6]}")
allj = np.unique(np.concatenate([a.ravel().astype(np.float64) for a in J.values() if a.dtype.kind == "f"]))
m = mace_off(model="small", default_dtype="float64", device="cpu").models[0]
print("torch model class:", type(m).__name__)
sd = {**dict(m.named_parameters()), **dict(m.named_buffers())}
bad, n, nexact = [], 0, 0
for k, t in sd.items():
    if not t.is_floating_point(): continue
    v = t.detach().double().numpy().ravel(); n += 1
    hit = np.isin(v, allj)
    if hit.all(): nexact += 1
    else:
        # close but not exact means float32 rounding or a transform; report the closest relative error
        idx = np.clip(np.searchsorted(allj, v[~hit]), 1, len(allj) - 1)
        near = np.minimum(np.abs(allj[idx] - v[~hit]), np.abs(allj[idx - 1] - v[~hit])) / np.maximum(np.abs(v[~hit]), 1e-300)
        bad.append((k, t.dtype, (~hit).sum(), v.size, near.max()))
print(f"floating torch tensors: {n}; found exactly (float64 bit-for-bit) in bundle: {nexact}")
for k, dt, nb, sz, r in bad: print(f"  not exact: {k} [{dt}] {nb}/{sz} values, max rel. distance to nearest bundle value {r:.1e}")
for name in ("atomic_energies_fn.atomic_energies", "scale_shift.scale", "scale_shift.shift"):
    if name in sd: print(f"{name}: torch {sd[name].dtype} {sd[name].detach().numpy().ravel()[:4]}")
for p, a in J.items():
    if any(s in p.lower() for s in ("atomic_energ", "scale", "shift")) and a.size < 20: print(f"bundle {p}: {a.dtype} {a.ravel()[:4]}")
