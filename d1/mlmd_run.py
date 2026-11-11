# D1 production driver: MLMD IR spectra with the Mai et al. 2025 models (NeuralIL force field + EPNN dipole,
# Zenodo 15771437) through the port in d1/port/mai_port.py, batched on one GPU.
#
# Protocol (the authors' code/1_MD_calc_position.py and 2_calc_dipole.py; validated in step 2, see
# docs/plan/cluster_migration.md): Maxwell-Boltzmann velocities at T (no momentum or rotation removal); NVT for
# nvt_steps at 0.5 fs with Langevin friction 0.02 per ASE time unit (BAOAB here, centre-of-mass velocity removed
# every step as ASE's fixcm); then NVE velocity Verlet for nve_steps; every 2 steps (1 fs) the EPNN dipole of the
# centre-of-mass-centred frame, positions rounded to 1e-5 A as in the authors' text trajectories.
# Several replicas (seeds) per species and temperature: one trajectory leaves small-PAH band ratios uncertain by
# tens of percent (step 2b).
#
# A work unit is up to --batch systems (species x T x replica) with the same atom count, run together (vmap):
#   python mlmd_run.py plan   <xyz dir> <outdir> [--temps 50,300,600] [--replicas 4] [--batch 100] [--nvt 400000] [--nve 400000]
#   python mlmd_run.py run    <Supplementary_Data_Code dir> <xyz dir> <outdir> [--shard k --nshards n] [--hours h]
#   python mlmd_run.py status <outdir>
# run takes the units with index % nshards == shard (units are sorted by cost, so shards balance), resumes from
# checkpoints, and stops cleanly after --hours (exit 0; rerun to continue). Per unit, <outdir>/units/<id>/ holds
# meta.json (systems, settings, QA flags), dipole.npy (n_frames, B, 3) float32 in e*A, state.npz (checkpoint, also
# holding QA: NVE E_tot every 100 frames, temperatures at every chunk end) and DONE when finished.
import argparse, glob, json, os, sys, time, zlib
import numpy as np

CHUNK = 2000            # MD steps per jitted call (even)
QA_EVERY = 100          # frames between E_tot / T samples
CKPT_SECONDS = 600      # checkpoint at least this often
DIPOLE_ROUND = 1e-5     # A; the authors wrote positions with %.5f
EDGE_CAP_FACTOR = 1.5   # dipole-graph edge cap = factor x the largest initial per-molecule edge count
SELFCHECK_F = 1e-3      # eV/A: forces with the vmapped descriptor vs the authors' lax.map one

# The authors' PowerSpectrumGenerator.process_data loops over atom centres with jax.lax.map, so a step costs time
# proportional to the atom count whatever the batch (step 4: t = n (a + b B), a = 0.30 s/ps per atom). The same
# per-centre function under jax.vmap gives the same descriptors, in parallel. _DESC["vmap"] is read at trace time:
# the engine traces with it on; Port.dipole (the self-check reference) and forces_ref trace with it off.
_DESC = {"vmap": True}


def _patch_descriptors():
    import jax
    from neuralil.bessel_descriptors import PowerSpectrumGenerator as PSG
    if getattr(PSG, "_mlmd_patched", False):
        return
    orig = PSG.process_data

    def process_data(self, coordinates, a_types, cell_size):
        if not _DESC["vmap"]:
            return orig(self, coordinates, a_types, cell_size)
        deltas, radii, a_types = self.center_at_atoms(coordinates, a_types, cell_size)
        weights = jax.nn.one_hot(a_types, self._n_types)
        return jax.vmap(jax.checkpoint(lambda d, r: self._process_center(d, r, a_types, weights)))(deltas, radii)
    PSG.process_data = process_data
    PSG._mlmd_patched = True
    # jnp.repeat with array repeats builds its gather indices with cumsum/scatter ops on constants; under the vmap
    # those become B x n x 25 x 25 x ... constant arrays that XLA folds on the CPU for many minutes (_process_center
    # repeats by self._degeneracies). Same result, with the indices computed once in NumPy.
    import jax.numpy as jnp
    _repeat = jnp.repeat

    def repeat(a, repeats, axis=None, *, total_repeat_length=None, **kw):
        if total_repeat_length is None and not kw and np.ndim(repeats) == 1 and not isinstance(repeats, jax.core.Tracer):
            a = jnp.asarray(a)
            if axis is None:
                a, axis = a.ravel(), 0
            r = np.broadcast_to(np.asarray(repeats), (a.shape[axis],))
            return jnp.take(a, np.repeat(np.arange(a.shape[axis]), r), axis=axis)
        return _repeat(a, repeats, axis, total_repeat_length=total_repeat_length, **kw)
    jnp.repeat = repeat


def natoms(path):
    with open(path) as f:
        return int(f.readline())


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


# ---------------------------------------------------------------------------------------------------------- plan
def cmd_plan(a):
    files = sorted(glob.glob(f"{a.xyz}/*.xyz"))
    temps = [float(t) for t in a.temps.split(",")]
    by_n = {}
    for f in files:
        sp = os.path.basename(f)[:-4]
        for T in temps:
            for r in range(a.replicas):
                by_n.setdefault(natoms(f), []).append({"species": sp, "T": T, "replica": r, "seed": seed_of(sp, T, r)})
    units = []
    for n, systems in sorted(by_n.items()):
        k = -(-len(systems) // a.batch)
        for j, part in enumerate(np.array_split(np.arange(len(systems)), k)):
            units.append({"id": f"n{n:03d}_u{j:03d}", "n_atoms": n, "systems": [systems[i] for i in part],
                          "cost": float(n * len(part))})
    units.sort(key=lambda u: -u["cost"])
    plan = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "xyz_dir": os.path.abspath(a.xyz),
            "settings": {"temps": temps, "replicas": a.replicas, "batch": a.batch, "nvt_steps": a.nvt,
                         "nve_steps": a.nve, "dt_fs": 0.5, "friction_ase": 2e-2, "frame_every": 2,
                         "chunk": CHUNK, "dipole_round_A": DIPOLE_ROUND},
            "units": units}
    assert a.nvt % CHUNK == 0 and a.nve % CHUNK == 0 and a.nve > 0, f"step counts must be multiples of {CHUNK}"
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(f"{a.out}/plan.json") and not a.force:
        sys.exit(f"{a.out}/plan.json exists; use --force to overwrite (running units would no longer match)")
    with open(f"{a.out}/plan.json", "w") as f:
        json.dump(plan, f, indent=1)
    nsys = sum(len(u["systems"]) for u in units)
    print(f"{len(files)} species x {len(temps)} T x {a.replicas} replicas = {nsys} systems in {len(units)} units "
          f"({len(by_n)} atom counts); plan in {a.out}/plan.json")


# ----------------------------------------------------------------------------------------------------------- run
class Engine:
    """Model functions, jitted once per input shape; per-unit arrays are arguments, not constants."""

    def __init__(self, root, cache_dir):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "port"))
        from mai_port import Port, MASS, MAX_NEIGHBORS, _patch
        import jax
        _patch(root)
        _patch_descriptors()                    # before Port() binds process_data into the models
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 10)
        import jax.numpy as jnp, jraph
        from ase import units
        self.jax, self.jnp, self.units, self.MASS = jax, jnp, units, MASS
        self.m = m = Port(root)                 # puts the authors' libraries on sys.path
        from neuralil.bessel_descriptors import center_at_atoms
        from epnn.preprocessing import smooth_cutoff
        self.calc = calc = m.calculator(n_devices=1)
        self.symbol_map, self.d_map = calc.symbol_map, m.d_map
        self.r_nnff, self.max_nb = float(m.ff.r_cut), MAX_NEIGHBORS
        self.dt = 0.5 * units.fs
        params, ens = calc.params, m.ensemble
        dp, epnn = m.dp, m.epnn
        r_cut, r_switch, nsd = float(dp.r_cut), float(dp.r_switch), int(dp.node_state_dim)
        cell0 = jnp.zeros((3, 3), jnp.float32)
        # the engine's own, unjitted EPNN descriptor: m.d_desc is jitted, and its trace cache must stay the lax.map one
        from neuralil.bessel_descriptors import PowerSpectrumGenerator
        d_desc = PowerSpectrumGenerator(6, dp.r_cut, len(m.d_elems), MAX_NEIGHBORS).process_data

        def energy(p, t):                       # the calculator's function: ensemble mean, summed over atoms
            return ens.apply(params, p, t, cell0, method=ens.calc_potential_energy).sum() / ens.n_models
        vef = jax.vmap(jax.value_and_grad(energy))

        def forces(p, t):
            e, g = vef(p, t)
            return e, -g

        def com_free(v, mass):
            return v - (mass * v).sum(1, keepdims=True) / mass.sum(1, keepdims=True)

        def ekin(v, mass):
            return 0.5 * (mass * v ** 2).sum((1, 2))

        def edges_and_dist(pc, dt_):
            dist = jax.vmap(center_at_atoms)(pc, dt_, jnp.zeros((pc.shape[0], 3, 3), jnp.float32))[1]
            return dist, (dist > 0) & (dist < r_cut)

        def charges(p, dt_, w, e_cap):
            """EPNN charges of B frames, as create_batched_graph_list + jraph padding (one padding graph), but with a
            static edge count so it traces; exact edge set and order (0 < d < r_cut, row-major over (b, i, j))."""
            B, n = dt_.shape
            p = jnp.round(p / DIPOLE_ROUND) * DIPOLE_ROUND
            pc = p - (w[..., None] * p).sum(1, keepdims=True) / w.sum(1)[:, None, None]
            desc = jax.vmap(d_desc)(pc, dt_, jnp.zeros((B, 3, 3), jnp.float32))
            dd = int(np.prod(desc.shape[2:]))
            dist, mask = edges_and_dist(pc, dt_)
            cnt = mask.sum()
            idx = jnp.nonzero(mask.ravel(), size=e_cap, fill_value=0)[0]
            valid = jnp.arange(e_cap) < cnt
            b, i, j = idx // (n * n), (idx // n) % n, idx % n
            pad = B * n
            d = dist.ravel()[idx]
            edges = jnp.where(valid[:, None], jnp.stack([d, smooth_cutoff(d, r_switch, r_cut)], -1), 0.0)
            g = jraph.GraphsTuple(
                nodes={"features": jnp.zeros((pad + 1, 1), jnp.float32),
                       "descriptors": jnp.concatenate([desc.reshape(pad, -1), jnp.zeros((1, dd), jnp.float32)]),
                       "types": jnp.concatenate([dt_.ravel(), jnp.zeros(1, dt_.dtype)]),
                       "node_states": jnp.zeros((pad + 1, nsd), jnp.float32)},
                senders=jnp.where(valid, b * n + i, pad), receivers=jnp.where(valid, b * n + j, pad),
                edges=edges,
                n_node=jnp.concatenate([jnp.full(B, n, jnp.int32), jnp.ones(1, jnp.int32)]),
                n_edge=jnp.concatenate([mask.reshape(B, -1).sum(1), (e_cap - cnt)[None]]).astype(jnp.int32),
                globals=None)
            q = epnn.apply(dp.params, g).nodes["features"][:pad, 0].reshape(B, n)
            return q, pc, cnt

        def dipole(p, dt_, w, e_cap):
            q, pc, cnt = charges(p, dt_, w, e_cap)
            return (q[..., None] * pc).sum(1), cnt

        def max_neighbours(p):                  # NNFF descriptor limit (the calculator raises above it)
            d = jnp.linalg.norm(p[:, :, None] - p[:, None], axis=-1)
            return ((d > 0) & (d < self.r_nnff)).sum(-1).max(-1)

        dt = self.dt

        def vv(state, t, mass):
            p, v, f = state
            v = v + 0.5 * dt * f / mass
            p = p + dt * v
            e, f = forces(p, t)
            v = v + 0.5 * dt * f / mass
            return (p, v, f), e

        def nvt_chunk(state, seeds, chunk, t, mass, kT, c1, c2):
            # noise keyed per system (seed, chunk): a trajectory does not depend on its batch or on resumes
            keys = jax.vmap(lambda sd: jax.random.split(jax.random.fold_in(jax.random.PRNGKey(sd), chunk), CHUNK))(seeds)
            keys = jnp.swapaxes(keys, 0, 1)                     # (CHUNK, B, key)
            def step(s, k):
                p, v, f = s
                v = v + 0.5 * dt * f / mass
                p = p + 0.5 * dt * v
                noise = jax.vmap(lambda kk: jax.random.normal(kk, v.shape[1:]))(k)
                v = c1 * v + c2 * jnp.sqrt(kT[:, None, None] / mass) * noise
                v = com_free(v, mass)
                p = p + 0.5 * dt * v
                e, f = forces(p, t)
                v = v + 0.5 * dt * f / mass
                return (p, v, f), None
            s, _ = jax.lax.scan(step, state, keys)
            return s

        def nve_chunk(state, t, mass, dt_, w, e_cap):
            def frame(s, _):
                s, _e = vv(s, t, mass)
                s, e = vv(s, t, mass)
                mu, cnt = dipole(s[0], dt_, w, e_cap)
                return s, (mu, e + ekin(s[1], mass), cnt)
            s, (mu, etot, cnt) = jax.lax.scan(frame, state, None, length=CHUNK // 2)
            return s, mu, etot, cnt.max()

        self.forces = jax.jit(forces)
        _fr = jax.jit(lambda p, t: forces(p, t)[1])

        def forces_ref(p, t):                   # the authors' lax.map descriptor (traced with the switch off)
            _DESC["vmap"] = False
            try:
                return _fr(p, t)
            finally:
                _DESC["vmap"] = True
        self.forces_ref = forces_ref
        self.dipole = jax.jit(dipole, static_argnums=3)
        self.charges = jax.jit(charges, static_argnums=3)
        self.nvt_chunk = jax.jit(nvt_chunk)
        self.nve_chunk = jax.jit(nve_chunk, static_argnums=5)
        self.ekin = jax.jit(ekin)
        self.max_neighbours = jax.jit(max_neighbours)
        self.edges_and_dist = jax.jit(edges_and_dist)


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


SELFCHECK_Q = 1e-4      # e, per-atom charge: in-loop graph builder vs the authors' per-frame path


def selfcheck(E, p, t, dt_, w, e_cap, atoms, systems):
    """In-loop EPNN charges vs the authors' create_batched_graph_list path, for one system per species (at most 8).
    Charges, not dipoles, are gated: their noise does not grow with molecule size."""
    jnp = E.jnp
    first = {}
    for i, s in enumerate(systems):
        first.setdefault(s["species"], i)
    idx = list(first.values())[:8]
    q_b, _, _ = E.charges(p, dt_, w, e_cap)
    pr = np.asarray(jnp.round(p[jnp.asarray(idx)] / DIPOLE_ROUND) * DIPOLE_ROUND)   # rounded as in the loop (float32)
    dq = dmu = 0.0
    for k, i in enumerate(idx):
        _DESC["vmap"] = False
        try:
            mu1, q1 = E.m.dipole(atoms[i].__class__(atoms[i].symbols, positions=pr[k]))
        finally:
            _DESC["vmap"] = True
        dq = max(dq, float(np.abs(np.asarray(q_b[i]) - q1).max()))
        w_ = np.asarray(w[i], np.float64); pc = pr[k] - (w_[:, None] * pr[k]).sum(0) / w_.sum()
        dmu = max(dmu, float(np.abs((np.asarray(q_b[i])[:, None] * pc).sum(0) - mu1).max()))
    ii = jnp.asarray(idx)
    df = float(jnp.abs(E.forces(p[ii], t[ii])[1] - E.forces_ref(p[ii], t[ii])).max())
    return dq, dmu, df


def save_atomic(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def run_unit(E, plan, xyz, u, udir, deadline, stop_after):
    jax, jnp, units = E.jax, E.jnp, E.units
    from ase.io import read
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    S = plan["settings"]
    n_nvt, n_nve = S["nvt_steps"] // CHUNK, S["nve_steps"] // CHUNK
    n_frames = S["nve_steps"] // S["frame_every"]
    systems = u["systems"]; B, n = len(systems), u["n_atoms"]
    cache = {}
    for s in systems:
        if s["species"] not in cache:
            cache[s["species"]] = read(f"{xyz}/{s['species']}.xyz")
    atoms = [cache[s["species"]] for s in systems]
    t = jnp.asarray([[E.symbol_map[x] for x in a.get_chemical_symbols()] for a in atoms])
    dt_ = jnp.asarray([[E.d_map[x] for x in a.get_chemical_symbols()] for a in atoms])
    mass = jnp.asarray(np.array([a.get_masses() for a in atoms]), jnp.float32)[..., None]
    w = jnp.asarray(np.array([[E.MASS[x] for x in a.get_chemical_symbols()] for a in atoms]), jnp.float32)
    T = np.array([s["T"] for s in systems]); kT = jnp.asarray(units.kB * T, jnp.float32)
    c1 = float(np.exp(-S["friction_ase"] * E.dt)); c2 = float(np.sqrt(1 - c1 ** 2))
    p0 = jnp.asarray(np.array([a.positions for a in atoms]), jnp.float32)

    os.makedirs(udir, exist_ok=True)
    meta_path, state_path = f"{udir}/meta.json", f"{udir}/state.npz"
    # dipole-graph edge cap from the initial geometries, checked on every frame
    _, mask0 = E.edges_and_dist(p0, dt_)
    e_cap = int(np.ceil(EDGE_CAP_FACTOR * int(np.asarray(mask0.reshape(B, -1).sum(1)).max()) * B / 1024)) * 1024
    if os.path.exists(state_path):
        ck = np.load(state_path)
        p, v, f = (jnp.asarray(ck[k]) for k in ("p", "v", "f"))
        chunk, nvt_T, qa_etot, qa_T = int(ck["chunk"]), list(ck["nvt_T"]), list(ck["qa_etot"]), list(ck["qa_T"])
        flags = json.load(open(meta_path))["flags"]
        e_cap = json.load(open(meta_path))["edge_cap"]
        print(f"  resume {u['id']} at chunk {chunk}/{n_nvt + n_nve}", flush=True)
    else:
        V = []
        for s, a in zip(systems, atoms):
            b = a.copy()
            MaxwellBoltzmannDistribution(b, temperature_K=s["T"], rng=np.random.default_rng(s["seed"]))
            V.append(b.get_velocities())
        p, v = p0, jnp.asarray(np.array(V), jnp.float32)
        f = E.forces(p, t)[1]
        chunk, nvt_T, qa_etot, qa_T = 0, [], [], []
        flags = {"nonfinite": [False] * B, "hot": [False] * B,
                 "max_neighbours": np.asarray(E.max_neighbours(p0)).tolist(), "edge_overflow": False}
        dq, dmu, df = selfcheck(E, p0, t, dt_, w, e_cap, atoms, systems)
        assert dq < SELFCHECK_Q, f"{u['id']}: in-loop EPNN charges differ from the per-frame path by {dq:.2e} e"
        assert df < SELFCHECK_F, f"{u['id']}: vmapped-descriptor forces differ from lax.map ones by {df:.2e} eV/A"
        write_json(meta_path, {"id": u["id"], "n_atoms": n, "systems": systems, "settings": S, "edge_cap": e_cap,
                               "selfcheck": {"initial": {"max_dq": dq, "max_dmu": dmu, "max_df": df}}, "flags": flags})
    mu_file = np.lib.format.open_memmap(f"{udir}/dipole.npy", mode="r+" if os.path.exists(f"{udir}/dipole.npy") else "w+",
                                        dtype=np.float32, shape=(n_frames, B, 3))
    seeds = jnp.asarray(np.array([s["seed"] for s in systems], np.uint32))
    state = (p, v, f)
    t_ckpt, t0, done_here = time.time(), time.time(), 0
    chunk_s = []                                # (chunk, wall seconds) this session; the first of each kind includes compile

    def checkpoint():
        mu_file.flush()
        p, v, f = state
        save_atomic(state_path, p=np.asarray(p), v=np.asarray(v), f=np.asarray(f), chunk=chunk, nvt_T=np.array(nvt_T),
                    qa_etot=np.array(qa_etot), qa_T=np.array(qa_T))
        m = json.load(open(meta_path)); m["flags"] = flags
        m.setdefault("chunk_seconds", []).extend(chunk_s); chunk_s.clear(); write_json(meta_path, m)

    while chunk < n_nvt + n_nve:
        if time.time() > deadline or (stop_after is not None and done_here >= stop_after):
            checkpoint(); return False
        tc = time.time()
        if chunk < n_nvt:
            state = E.nvt_chunk(state, seeds, chunk, t, mass, kT, c1, c2)
            nvt_T.append(np.asarray(E.ekin(state[1], mass)) / (1.5 * n * units.kB))
        else:
            if chunk == n_nvt:                  # repeat the self-check on thermal frames (end of NVT)
                dq, dmu, df = selfcheck(E, state[0], t, dt_, w, e_cap, atoms, systems)
                m_ = json.load(open(meta_path)); m_["selfcheck"]["end_of_nvt"] = {"max_dq": dq, "max_dmu": dmu, "max_df": df}
                write_json(meta_path, m_)
                assert df < SELFCHECK_F, f"{u['id']}: vmapped-descriptor forces differ from lax.map ones by {df:.2e} eV/A (end of NVT)"
                assert dq < SELFCHECK_Q, f"{u['id']}: in-loop EPNN charges differ from the per-frame path by {dq:.2e} e (end of NVT)"
            state, mu, etot, cnt = E.nve_chunk(state, t, mass, dt_, w, e_cap)
            j0 = (chunk - n_nvt) * (CHUNK // 2)
            mu_file[j0:j0 + CHUNK // 2] = np.asarray(mu)
            etot = np.asarray(etot)
            qa_etot.extend(etot[::QA_EVERY])
            if int(cnt) > e_cap:
                flags["edge_overflow"] = True
        pk, vk = np.asarray(state[0]), np.asarray(state[1])
        Tk = np.asarray(E.ekin(state[1], mass)) / (1.5 * n * units.kB)
        if chunk >= n_nvt:
            qa_T.append(Tk)
        bad = ~np.isfinite(pk).all((1, 2)) | ~np.isfinite(vk).all((1, 2))
        flags["nonfinite"] = [x or bool(y) for x, y in zip(flags["nonfinite"], bad)]
        flags["hot"] = [x or bool(y) for x, y in zip(flags["hot"], np.abs(Tk) > 30000)]   # the authors' "dispersion" check
        flags["max_neighbours"] = np.maximum(flags["max_neighbours"], np.asarray(E.max_neighbours(state[0]))).tolist()
        chunk_s.append((chunk, round(time.time() - tc, 3)))
        chunk += 1; done_here += 1
        if time.time() - t_ckpt > CKPT_SECONDS:
            checkpoint(); t_ckpt = time.time()
    checkpoint()
    m = json.load(open(meta_path))
    m["seconds_last_session"] = time.time() - t0
    write_json(meta_path, m)
    open(f"{udir}/DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    nbad = sum(flags["nonfinite"]) + sum(flags["hot"]) + sum(x > E.max_nb for x in flags["max_neighbours"])
    print(f"  done {u['id']}: {B} systems x {n} atoms; flagged systems {nbad}; edge overflow {flags['edge_overflow']}", flush=True)
    return True


def cmd_run(a):
    plan = json.load(open(f"{a.out}/plan.json"))
    units_ = [u for u in plan["units"] if not a.units or u["id"] in a.units.split(",")]
    units_ = [u for i, u in enumerate(units_) if i % a.nshards == a.shard]
    deadline = time.time() + a.hours * 3600
    E = Engine(a.root, os.path.abspath(f"{a.out}/jax_cache"))
    print(f"jax {E.jax.__version__} {E.jax.devices()}; shard {a.shard}/{a.nshards}: {len(units_)} units", flush=True)
    for u in units_:
        udir = f"{a.out}/units/{u['id']}"
        if os.path.exists(f"{udir}/DONE"):
            continue
        t0 = time.time()
        finished = run_unit(E, plan, a.xyz, u, udir, deadline, a.stop_after)
        print(f"  {u['id']}: {'finished' if finished else 'paused'} after {time.time() - t0:.0f} s this session", flush=True)
        if not finished:
            break


def cmd_status(a):
    plan = json.load(open(f"{a.out}/plan.json"))
    S = plan["settings"]; total = (S["nvt_steps"] + S["nve_steps"]) // CHUNK
    done = started = 0; flagged = []
    for u in plan["units"]:
        d = f"{a.out}/units/{u['id']}"
        if os.path.exists(f"{d}/DONE"):
            done += 1
        elif os.path.exists(f"{d}/state.npz"):
            started += 1
        if os.path.exists(f"{d}/meta.json"):
            fl = json.load(open(f"{d}/meta.json"))["flags"]
            for k, s in enumerate(u["systems"]):
                if fl["nonfinite"][k] or fl["hot"][k] or fl["max_neighbours"][k] > 25:
                    flagged.append((u["id"], s["species"], s["T"], s["replica"]))
            if fl["edge_overflow"]:
                flagged.append((u["id"], "edge overflow"))
    print(f"{done}/{len(plan['units'])} units done, {started} in progress ({total} chunks each); {len(flagged)} flagged")
    for x in flagged[:50]:
        print("   ", x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("xyz"); p.add_argument("out")
    p.add_argument("--temps", default="50,300,600"); p.add_argument("--replicas", type=int, default=4)
    p.add_argument("--batch", type=int, default=100); p.add_argument("--nvt", type=int, default=400000)
    p.add_argument("--nve", type=int, default=400000); p.add_argument("--force", action="store_true")
    r = sub.add_parser("run"); r.add_argument("root"); r.add_argument("xyz"); r.add_argument("out")
    r.add_argument("--shard", type=int, default=0); r.add_argument("--nshards", type=int, default=1)
    r.add_argument("--hours", type=float, default=23.5)
    r.add_argument("--stop-after", type=int, default=None, help="stop after this many chunks (testing)")
    r.add_argument("--units", default=None, help="comma-separated unit ids: run only these (still split across shards); for calibration")
    s = sub.add_parser("status"); s.add_argument("out")
    a = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
