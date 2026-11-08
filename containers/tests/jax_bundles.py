# From the x64 master bundle (mace-jax-from-torch under JAX_ENABLE_X64=1), write:
#   off_small_x64_cueq: same parameters, cuEquivariance backend on
#   off_small_f32{,_cueq}: every floating leaf cast to float32 (benchmark precision), cueq off/on
import json, numpy as np
from flax import serialization
src = "/t/off_small_x64"
P = serialization.msgpack_restore(open(src + ".npz", "rb").read()); C = json.load(open(src + ".json"))
def cast(d):
    if isinstance(d, dict): return {k: cast(v) for k, v in d.items()}
    a = np.asarray(d); return a.astype(np.float32) if a.dtype.kind == "f" else a
def kinds(d, out):
    if isinstance(d, dict): [kinds(v, out) for v in d.values()]
    else: out.add(str(np.asarray(d).dtype))
    return out
Pc = cast(P)
for name, params in (("x64", P), ("f32", Pc)):
    for cq in ("", "_cueq"):
        c = dict(C)
        if cq: c["cueq_config"] = None; c["equivariance_config"] = {"backend": "cueq"}
        open(f"/t/off_small_{name}{cq}.npz", "wb").write(serialization.to_bytes(params))
        json.dump(c, open(f"/t/off_small_{name}{cq}.json", "w"))
        print(f"off_small_{name}{cq}: leaf dtypes {sorted(kinds(params, set()))}, equivariance_config {c.get('equivariance_config')}")
