# Time one jitted energy+forces call of a mace-jax bundle on a padded batch of molecules (default: the first
# 100 frames of eq_frames.xyz), after warm-up, with block_until_ready. Not the T0 benchmark: a quick check of
# whether a bundle variant changes the force-call cost. Usage: jax_time.py bundle.npz dtype [n_mols] [reps]
import sys, time, numpy as np, jax, jraph
from mace_jax.tools.bundle import load_model_bundle
from mace_jax.cli import mace_jax_predict as P
from mace_jax import data
bundle_path, dtype = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 100; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 20
b = load_model_bundle(bundle_path, dtype)
cfg = b.config
head_name, head_to_index = P._resolve_head_mapping(cfg, None)
_, configs = data.load_from_xyz("/t/eq_frames.xyz", head_name=head_name, no_data_ok=True)
z = data.AtomicNumberTable(cfg["atomic_numbers"])
gs = [data.graph_from_configuration(c, cutoff=float(cfg["r_max"]), z_table=z, head_to_index=head_to_index) for c in configs[:n]]
g = jraph.batch_np(gs)
nn, ne = int(g.n_node.sum()), int(g.n_edge.sum())
g = jraph.pad_with_graphs(g, n_node=nn + 1, n_edge=ne, n_graph=len(gs) + 1)
pred = jax.jit(P._build_predictor(b.graphdef, model_config=cfg, compute_forces=True, compute_stress=False, full_outputs=False))
t0 = time.perf_counter(); out = jax.block_until_ready(pred(b.params, g)); tc = time.perf_counter() - t0
ts = []
for _ in range(reps):
    t0 = time.perf_counter(); jax.block_until_ready(pred(b.params, g)); ts.append(time.perf_counter() - t0)
print(f"{bundle_path.split('/')[-1]} {dtype}: {len(gs)} mols, {nn} atoms, {ne} edges; compile+first {tc:.1f} s; "
      f"call median {1e3*np.median(ts):.2f} ms (min {1e3*min(ts):.2f}); energy dtype {out['energy'].dtype}")
