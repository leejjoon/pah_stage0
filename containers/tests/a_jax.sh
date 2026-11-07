set -e
cd /t
python -c "import jax; print('jax', jax.__version__, jax.devices(), 'x64', jax.config.jax_enable_x64)"
python -c "import torch; print('torch in A:', torch.__version__)"
python -c "import cuequivariance_ops_jax; print('cueq-ops-jax import ok without jax init')"
mace-jax-from-torch --foundation off --model-name small --output /t/ctr_off_small.npz 2>&1 | grep -iE "written|error" 
sha256sum $HOME/.cache/mace/MACE-OFF23_small.model
python -c "
import json; c=json.load(open('/t/ctr_off_small.json')); c['cueq_config']=None; c['equivariance_config']={'backend':'cueq'}; json.dump(c,open('/t/ctr_off_small_cueq.json','w'))"
cp /t/ctr_off_small.npz /t/ctr_off_small_cueq.npz
mace-jax-predict /t/ctr_off_small.npz /t/benzene.xyz --dtype float32 --no-progress --output /t/ctr_jax.npz 2>&1 | grep -iE "wrote|error"
mace-jax-predict /t/ctr_off_small_cueq.npz /t/benzene.xyz --dtype float32 --no-progress --output /t/ctr_jax_cueq.npz 2>&1 | grep -iE "wrote|error"
python - <<'PY'
import jax, jax.numpy as jnp, time
from jax_md import space, energy, simulate
disp, shift = space.periodic(12.0); R = jax.random.uniform(jax.random.PRNGKey(0),(500,3))*12.0
init, apply = simulate.nvt_langevin(energy.lennard_jones_pair(disp), shift, dt=1e-3, kT=1.0, gamma=0.1)
st = init(jax.random.PRNGKey(1), R); st = jax.block_until_ready(jax.jit(lambda s: jax.lax.fori_loop(0,1000,lambda i,s: apply(s), s))(st))
print("jax-md 1000 Langevin steps ok on", st.position.devices())
PY
