# mace-jax-predict with the cuEquivariance-JAX entry points wrapped, to print the effective math dtype and
# operand dtypes of every cuex call actually traced. Usage: same arguments as mace-jax-predict.
import sys, collections
import cuequivariance_jax as cuex
seen = collections.Counter()
def wrap(name):
    f = getattr(cuex, name)
    def g(*a, **k):
        ops = a[1] if len(a) > 1 else k.get("inputs", [])
        dts = tuple(sorted({str(getattr(getattr(o, "array", o), "dtype", "?")) for o in ops}))
        seen[(name, k.get("method"), str(k.get("math_dtype")), dts)] += 1
        return f(*a, **k)
    setattr(cuex, name, g)
for n in ("equivariant_polynomial", "segmented_polynomial"): wrap(n)
from mace_jax.cli import mace_jax_predict
import jax
sys.argv = ["mace-jax-predict", *sys.argv[1:]]
try:
    mace_jax_predict.main()
finally:
    print("jax_enable_x64 at exit:", jax.config.jax_enable_x64)
    if not seen: print("cuex calls traced: none (plain e3nn-jax path)")
    for (n, m, md, dts), c in sorted(seen.items()): print(f"cuex call traced: {n} method={m} math_dtype={md} operand dtypes={dts} x{c}")
