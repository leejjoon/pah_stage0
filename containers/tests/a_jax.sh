set -e
cd /t
python -c "import jax; print('jax', jax.__version__, jax.devices(), 'x64', jax.config.jax_enable_x64)"
python -c "import torch; print('torch in A:', torch.__version__)"
python -c "import cuequivariance_ops_jax; print('cueq-ops-jax import ok without jax init')"
# Plan 4.2: convert once under x64 (otherwise the float64 checkpoint is rounded to float32), verify against the
# torch checkpoint, keep it as the float64 master and derive float32 copies (cueq off/on) from it.
JAX_ENABLE_X64=1 mace-jax-from-torch --foundation off --model-name small --output /t/off_small_x64.npz 2>&1 | grep -iE "written|error"
sha256sum $HOME/.cache/mace/MACE-OFF23_small.model
python /t/x64_verify.py /t/off_small_x64.npz 2>&1 | grep -vi warn
python /t/jax_bundles.py
mace-jax-predict /t/off_small_f32.npz /t/benzene.xyz --dtype float32 --no-progress --output /t/ctr_jax.npz 2>&1 | grep -iE "wrote|error"
mace-jax-predict /t/off_small_f32_cueq.npz /t/benzene.xyz --dtype float32 --no-progress --output /t/ctr_jax_cueq.npz 2>&1 | grep -iE "wrote|error"
python - <<'PY'
import jax, jax.numpy as jnp, time
from jax_md import space, energy, simulate
disp, shift = space.periodic(12.0); R = jax.random.uniform(jax.random.PRNGKey(0),(500,3))*12.0
init, apply = simulate.nvt_langevin(energy.lennard_jones_pair(disp), shift, dt=1e-3, kT=1.0, gamma=0.1)
st = init(jax.random.PRNGKey(1), R); st = jax.block_until_ready(jax.jit(lambda s: jax.lax.fori_loop(0,1000,lambda i,s: apply(s), s))(st))
print("jax-md 1000 Langevin steps ok on", st.position.devices())
PY
