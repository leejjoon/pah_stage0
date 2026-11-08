"""T0 JAX driver (plan 4.3): jax-md BAOAB Langevin (or velocity-Verlet NVE) on one padded jraph graph holding all
b molecules, with mace-jax forces. Run in container A, pinned:
  CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 apptainer exec --nv --bind <repo>:/r A_jax.sif \
    python /r/t0/md_jax.py --bundle <off_small_f32[_cueqf].npz> --batch 100 --out <prefix> [--check]

Design (plan 4.3):
- Integrator: jax_md.simulate's BAOAB building blocks (momentum_step, position_step, stochastic_step) in the same
  order as simulate.nvt_langevin's step_fn, so that energy and forces come from ONE model call per step (nvt_langevin
  keeps only the force). --check verifies step-for-step agreement with nvt_langevin itself.
- Units: Angstrom, amu, eV; one time unit = 10.18 fs. Masses passed explicitly. kT in eV.
- Edges: rebuilt every step inside the jitted chunk from the precomputed list of all intramolecular atom pairs,
  cut at r_max and packed into a fixed edge cap (jnp.nonzero with size=cap). The true count is returned per step;
  if it ever exceeds the cap, the chunk is re-run from its starting state with a larger cap (never truncated).
- Padding: one padding node (and the jraph padding graph); it has no edges, zero force and is excluded from every
  observable. Molecules never share edges, so no inter-molecular interaction exists by construction.
- COM: identical initial velocities from t0/common.py (no COM removal); no COM constraint in the step.
- Precision: float32 parameters, jax_enable_x64 off, JAX_DEFAULT_MATMUL_PRECISION=highest (asserted).
"""
import argparse, csv, importlib.metadata, json, os, sys, tempfile, time
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax, jax.numpy as jnp, jraph
import cuequivariance_jax as cuex

TRACED = set()
def _wrap(name):
    f = getattr(cuex, name)
    def g(*a, **k):
        TRACED.add(f"{name}:{k.get('method')}:{k.get('math_dtype')}"); return f(*a, **k)
    setattr(cuex, name, g)
for _n in ("equivariant_polynomial", "segmented_polynomial"): _wrap(_n)

from jax_md import space, simulate
from mace_jax import data
from mace_jax.cli import mace_jax_predict as MP
from mace_jax.tools import gin_model
from mace_jax.tools.bundle import load_model_bundle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", required=True)
ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
ap.add_argument("--xyz", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "species_T0.xyz"))
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--equil", type=int, default=C.EQUIL_STEPS)
ap.add_argument("--prod", type=int, default=C.PROD_STEPS)
ap.add_argument("--chunk", type=int, default=1000)
ap.add_argument("--ensemble", default="nvt", choices=["nvt", "nve"])
ap.add_argument("--save-vel", type=int, default=0, help="save per-step velocities of the first K trajectories (VDOS)")
ap.add_argument("--edge-slack", type=float, default=1.3)
ap.add_argument("--check", action="store_true", help="run the correctness checks and exit")
ap.add_argument("--out", default="md_jax")
ap.add_argument("--csv", default=None)
ap.add_argument("--repeat", type=int, default=0)
ap.add_argument("--seed", type=int, default=20261108)
args = ap.parse_args()

bundle = load_model_bundle(args.bundle, args.dtype)
cfg, params, graphdef = bundle.config, bundle.params, bundle.graphdef
if args.dtype == "float32":
    assert not jax.config.jax_enable_x64, "x64 must be off for the float32 benchmark"
    leaf_dt = {str(x.dtype) for x in jax.tree_util.tree_leaves(params) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)}
    assert leaf_dt == {"float32"}, f"float32 run with parameter dtypes {leaf_dt}"
assert os.environ["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"
rmax = float(cfg["r_max"]); ns = len(cfg["atomic_numbers"])
FT = jnp.float32 if args.dtype == "float32" else jnp.float64

# ---- the batch -----------------------------------------------------------------------------------------------
species = C.read_xyz(args.xyz)
mols = C.batch(species, args.batch); b = len(mols)
nat = np.array([len(z) for z, _ in mols]); N = int(nat.sum())
Z = np.concatenate([z for z, _ in mols]); X0 = np.concatenate([x for _, x in mols])
mol = np.repeat(np.arange(b), nat)
m = np.array([C.MASS[z] for z in Z])
q = np.concatenate([C.proxy_charges(z, x) for z, x in mols])
V0 = np.concatenate([C.initial_velocities(z, k, args.seed) for k, (z, _) in enumerate(mols)])   # Angstrom/fs
off = np.concatenate([[0], np.cumsum(nat)])
pi, pj, pg = [], [], []
for k in range(b):
    ii, jj = np.meshgrid(np.arange(off[k], off[k + 1]), np.arange(off[k], off[k + 1]), indexing="ij")
    sel = ii != jj; pi.append(ii[sel]); pj.append(jj[sel]); pg.append(np.full(sel.sum(), k))
pi, pj, pg = (np.concatenate(a).astype(np.int32) for a in (pi, pj, pg))

# template graph: per-molecule graphs from mace-jax's own builder, batched and padded with one node and one graph
def mace_graphs(mol_list):
    with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as f:
        for z, x in mol_list:
            f.write(f"{len(z)}\nProperties=species:S:1:pos:R:3 pbc=\"F F F\"\n")
            for zz, xx in zip(z, x): f.write(f"{'H' if zz == 1 else 'C'} {xx[0]:.10f} {xx[1]:.10f} {xx[2]:.10f}\n")
    head_name, head_to_index = MP._resolve_head_mapping(cfg, None)
    _, confs = data.load_from_xyz(f.name, head_name=head_name, no_data_ok=True); os.unlink(f.name)
    zt = data.AtomicNumberTable(cfg["atomic_numbers"])
    return [data.graph_from_configuration(c, cutoff=rmax, z_table=zt, head_to_index=head_to_index) for c in confs]

g_list = mace_graphs(mols)
exact_edges = int(sum(int(g.n_edge.sum()) for g in g_list))

def template(cap):
    g = jraph.batch_np(g_list)
    return jraph.pad_with_graphs(g, n_node=N + 1, n_edge=cap, n_graph=b + 1)

apply = graphdef.apply(params)

def model(graph):
    out, _ = apply(gin_model._graph_to_data(graph, num_species=ns), compute_force=True, compute_stress=False, compute_node_feats=False)
    return out

def make_ef(cap):
    tg = template(cap)
    tg = jax.tree_util.tree_map(lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, tg)
    PI, PJ, PG = jnp.asarray(pi), jnp.asarray(pj), jnp.asarray(pg)
    def ef(R):
        d = jnp.linalg.norm(R[PI] - R[PJ], axis=1)
        mask = d < rmax
        cnt = mask.sum()
        idx = jnp.nonzero(mask, size=cap, fill_value=-1)[0]
        ok = idx >= 0
        s = jnp.where(ok, PI[jnp.clip(idx, 0)], N); r = jnp.where(ok, PJ[jnp.clip(idx, 0)], N)
        ne = jax.ops.segment_sum(mask.astype(jnp.int32), PG, num_segments=b)
        ne = jnp.concatenate([ne, jnp.array([cap], jnp.int32) - ne.sum()[None]])
        graph = tg._replace(nodes=tg.nodes._replace(positions=R), senders=s.astype(tg.senders.dtype),
                            receivers=r.astype(tg.receivers.dtype), n_edge=ne.astype(tg.n_edge.dtype))
        out = model(graph)
        F = out["forces"].at[N].set(0.0)
        Eint = out.get("interaction_energy", out["energy"])
        return out["energy"][:b], Eint[:b], F, cnt
    return ef

# ---- state and step ------------------------------------------------------------------------------------------
dt = C.DT_FS / C.FS_PER_TU; gamma = C.GAMMA_PER_FS * C.FS_PER_TU; kT = C.KB * C.T_K
print(f"units: dt {dt:.5f} tu, gamma {gamma:.5f} /tu, gamma*dt {gamma * dt:.5f} (expect {C.GAMMA_PER_FS * C.DT_FS}), kT {kT:.5f} eV")
_, shift = space.free()
mass = jnp.asarray(np.append(m, 1.0)[:, None], FT)
P0 = jnp.asarray(np.vstack([m[:, None] * V0 * C.FS_PER_TU, np.zeros((1, 3))]), FT)    # amu Angstrom / tu
R0 = jnp.asarray(np.vstack([X0, np.zeros((1, 3))]), FT)
MOL = jnp.asarray(np.append(mol, b)); Q = jnp.asarray(np.append(q, 0.0), FT)
nsave = int(off[min(args.save_vel, b)])

def observe(st, E, Eint, cnt):
    ke_atom = 0.5 * (st.momentum ** 2 / st.mass).sum(1)
    ke = jax.ops.segment_sum(ke_atom, MOL, num_segments=b + 1)[:b]
    dip = jax.ops.segment_sum(Q[:, None] * st.position, MOL, num_segments=b + 1)[:b]
    o = dict(ke=ke, epot=E, eint=Eint, dip=dip, edges=cnt)
    if nsave: o["vel"] = (st.momentum[:nsave] / st.mass[:nsave]) / C.FS_PER_TU      # Angstrom/fs
    return o

def make_chunk(cap, nsteps):
    ef = make_ef(cap)
    def step(st, _):
        st = simulate.momentum_step(st, dt / 2)
        st = simulate.position_step(st, shift, dt / 2)
        if args.ensemble == "nvt": st = simulate.stochastic_step(st, dt, kT, gamma)
        st = simulate.position_step(st, shift, dt / 2)
        E, Eint, F, cnt = ef(st.position)
        st = st.set(force=F)
        st = simulate.momentum_step(st, dt / 2)
        return st, observe(st, E, Eint, cnt)
    return jax.jit(lambda st: jax.lax.scan(step, st, None, length=nsteps)), ef

def init_state(ef):
    E, Eint, F, cnt = ef(R0)
    if args.ensemble == "nvt":
        return simulate.NVTLangevinState(R0, P0, F, mass, jax.random.PRNGKey(args.seed)), int(cnt)
    return simulate.NVEState(R0, P0, F, mass), int(cnt)

cap = int(np.ceil(exact_edges * args.edge_slack))
print(f"batch {b}: {N} atoms, {len(pi)} candidate pairs, {exact_edges} edges at start (mace-jax builder), edge cap {cap}")

# ---- correctness checks ------------------------------------------------------------------------------------
if args.check:
    ef = make_ef(cap)
    E, Eint, F, cnt = jax.block_until_ready(ef(R0))
    print(f"edge count, dynamic builder vs mace-jax builder: {int(cnt)} vs {exact_edges}")
    # (a) batched, padded graph from the dynamic builder vs each molecule alone, unpadded, from mace-jax's builder
    dF, dE = [], []
    for k, g in enumerate(g_list):
        g1 = jraph.pad_with_graphs(g, n_node=int(g.n_node.sum()) + 1, n_edge=int(g.n_edge.sum()), n_graph=2)
        o = jax.jit(model)(g1)
        dF.append(float(jnp.abs(o["forces"][:nat[k]] - F[off[k]:off[k + 1]]).max())); dE.append(abs(float(o["energy"][0] - E[k])))
    print(f"batched+padded vs isolated unpadded: max|dF| {max(dF):.2e} eV/A, max|dE| {max(dE):.2e} eV (float32 totals ~1e4-1e5 eV)")
    print(f"padding node force {float(jnp.abs(F[N]).max()):.1e}")
    # (b) one step of this driver vs simulate.nvt_langevin's own step_fn (force-only function), same state and key
    if args.ensemble == "nvt":
        st, _ = init_state(ef)
        chunk1, _ = make_chunk(cap, 1)
        mine, _ = chunk1(st)
        init_fn, step_fn = simulate.nvt_langevin(lambda R: ef(R)[2], shift, dt, kT, gamma)
        ref = step_fn(st)
        print(f"one BAOAB step vs jax_md nvt_langevin: max|dR| {float(jnp.abs(mine.position - ref.position).max()):.1e} A, "
              f"max|dP| {float(jnp.abs(mine.momentum - ref.momentum).max()):.1e}")
    # (c) temperature of the initial velocities, 3N dof, padding excluded
    print(f"initial T (all atoms, 3N dof): {C.temperature(V0, m):.1f} K")
    print("cuex calls traced:", sorted(TRACED) or "none")
    sys.exit(0)

# ---- run ---------------------------------------------------------------------------------------------------
COMPILED, COMPILE_S = {}, {}
def get_chunk(cap, st):
    """One compiled executable per edge cap; compile time recorded separately and excluded from the timing."""
    if cap not in COMPILED:
        t0 = time.perf_counter()
        COMPILED[cap] = make_chunk(cap, args.chunk)[0].lower(st).compile()
        COMPILE_S[cap] = time.perf_counter() - t0
    return COMPILED[cap]

def run(st, nsteps):
    global cap
    chunk = get_chunk(cap, st)
    obs, t_wall, done = [], 0.0, 0
    while done < nsteps:
        t0 = time.perf_counter()
        st1, o = jax.block_until_ready(chunk(st))
        o = jax.tree_util.tree_map(np.asarray, o)
        dt_ = time.perf_counter() - t0
        if o["edges"].max() > cap:
            cap = int(np.ceil(o["edges"].max() * args.edge_slack))
            print(f"  edge cap overflow ({o['edges'].max()} edges): re-running the chunk with cap {cap}")
            chunk = get_chunk(cap, st); continue
        st = st1; obs.append(o); t_wall += dt_; done += args.chunk
    return st, ({k: np.concatenate([o[k] for o in obs]) for k in obs[0]} if obs else {}), t_wall

st, cnt0 = init_state(jax.jit(make_ef(cap)))
get_chunk(cap, st)
print(f"compile {COMPILE_S[cap]:.1f} s")
st, eq_obs, eq_wall = run(st, args.equil)
st, obs, wall = run(st, args.prod)
compile_s = sum(COMPILE_S.values())
peak = jax.devices()[0].memory_stats().get("peak_bytes_in_use", 0) / 1e9

mm = np.array([C.MASS[z] for z in Z]); dof = 3 * nat
T = 2 * obs["ke"] / (dof[None, :] * C.KB)
ps = args.prod * C.DT_FS / 1000
print(f"production {args.prod} steps ({ps:g} ps) x {b} molecules: {wall:.1f} s wall -> {wall / (ps * b):.3f} s per molecule-ps, "
      f"{86400 / (wall / ps) / 1000:.3f} ns/day per trajectory; peak GPU mem {peak:.2f} GB; edges {obs['edges'].min()}-{obs['edges'].max()} (cap {cap})")
print(f"T mean {T.mean():.1f} K (per-trajectory mean {T.mean(0).min():.0f}-{T.mean(0).max():.0f} K)")
if args.ensemble == "nve":
    etot = obs["eint"].astype(np.float64) + obs["ke"].astype(np.float64)
    drift = (etot[-1] - etot[0]) / nat
    print(f"NVE drift of interaction + kinetic energy: max |dE|/atom {np.abs(drift).max():.2e} eV over {ps:g} ps")
print("cuex calls traced:", sorted(TRACED) or "none")
np.savez_compressed(args.out + ".npz", T=T, epot=obs["epot"], eint=obs["eint"], dip=obs["dip"], edges=obs["edges"],
                    nat=nat, **({"vel": obs["vel"]} if nsave else {}))
if args.csv:
    cueq = (cfg.get("equivariance_config") or {}).get("backend") == "cueq"
    row = dict(stack="jax", mode="batched", batched_impl="jax", cueq="yes" if cueq else "no",
               cueq_version=importlib.metadata.version("cuequivariance"), conv_fusion=(cfg.get("equivariance_config") or {}).get("conv_fusion", False),
               dtype=args.dtype, matmul_precision=os.environ["JAX_DEFAULT_MATMUL_PRECISION"], batch=b, mps_workers="", repeat=args.repeat,
               species_set="T0_20", ensemble=args.ensemble, wall_s=round(wall, 3), s_per_mol_ps=round(wall / (ps * b), 5),
               ns_per_day_per_traj=round(86400 / (wall / ps) / 1000, 5), gpu_util="", gpu_mem_gb=round(peak, 3), compile_s=round(compile_s, 2),
               steps=args.prod, edge_cap=cap, cuex=";".join(sorted(TRACED)))
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
