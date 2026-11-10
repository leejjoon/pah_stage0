# Mai et al. 2025 MLMD models (Zenodo 15771437) in container A (JAX 0.11, Flax 0.12): the authors' own classes from
# lib/ of the Zenodo zip (Apache 2.0), patched only for APIs removed since their JAX 0.4.19 / Flax 0.7.4, and their
# pickles loaded with an allow-list unpickler. Shared by mai_port_test.py and mai_compare.py.
#   from mai_port import Port; m = Port("<Supplementary_Data_Code dir>")
#   m.calculator()  -> ASE calculator (NNFF ensemble, as in 1_MD_calc_position.py)
#   m.dipole(atoms) -> (dipole e*A, per-atom charges), as in 2_calc_dipole.py
import sys, pickle, importlib
import numpy as np

MAX_NEIGHBORS, N_ENS = 25, 5                   # 1_MD_calc_position.py and 2_calc_dipole.py
MASS = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}   # read_coord() in 2_calc_dipole.py

_patched = False
def _patch(root):
    """Put the authors' libraries on sys.path and patch JAX for the APIs they use that no longer exist."""
    global _patched
    if _patched: return
    sys.path[:0] = [f"{root}/lib/bessel-nn-potentials-velo/src", f"{root}/lib/epnn-main/src"]
    import jax, jax.numpy as jnp
    if not hasattr(jax, "tree_map"):                       # removed in JAX 0.6; the Mai code calls it
        jax.tree_map = jax.tree.map
    _clip = jnp.clip                                       # a_min/a_max were removed (now min/max); spherical_bessel uses a_max
    jnp.clip = lambda x, min=None, max=None, a_min=None, a_max=None, **kw: _clip(
        x, min if a_min is None else a_min, max if a_max is None else a_max, **kw)
    sys.modules.setdefault("jax.config", jax.config)       # the jax.config module is gone; bessel_descriptors imports it
    _update = jax.config.update                            # ase_integration_gai forces the CPU platform on import
    jax.config.update = lambda k, v: None if k == "jax_platform_name" else _update(k, v)
    import neuralil.ase_integration_gai                    # noqa: F401  (import while the platform switch is disabled)
    jax.config.update = _update
    _patched = True

class _Safe(pickle.Unpickler):
    ALLOWED = {("datetime", "datetime"), ("numpy", "ndarray"), ("numpy", "dtype"),
               ("neuralil.model", "NeuralILModelInfo"), ("epnn.model", "EPNNModelInfo")}
    def find_class(self, mod, name):
        if mod in ("numpy.core.multiarray", "numpy._core.multiarray") and name in ("_reconstruct", "scalar"):
            return getattr(importlib.import_module("numpy._core.multiarray"), name)
        if (mod, name) == ("jax._src.array", "_reconstruct_array"):
            return getattr(importlib.import_module(mod), name)
        if (mod, name) in self.ALLOWED:
            return getattr(importlib.import_module(mod), name)
        raise pickle.UnpicklingError(f"blocked global {mod}.{name}")

def load(p):
    with open(p, "rb") as f:
        return _Safe(f).load()

class Port:
    def __init__(self, root):
        _patch(root)
        import jax, jax.numpy as jnp
        from neuralil.bessel_descriptors import PowerSpectrumGenerator
        from neuralil.model_older import ResNetCore as ResNetCoreOld, NeuralILwithMorse
        from neuralil.model import ResNetCore
        from neuralil.plain_ensembles.model import PlainEnsemblewithMorse
        from epnn.model import EPNN, MLP
        self.ff = ff = load(f"{root}/model/model_enery.pkl")
        self.dp = dp = load(f"{root}/model/model_dipole.pkl")
        desc = PowerSpectrumGenerator(ff.n_max, ff.r_cut, len(ff.sorted_elements), MAX_NEIGHBORS)
        individual = NeuralILwithMorse(len(ff.sorted_elements), ff.embed_d, ff.r_cut, desc.process_data,
                                       desc.process_some_data, ResNetCoreOld(ff.core_widths))
        self.ensemble = PlainEnsemblewithMorse(individual, N_ENS)
        # EPNN, as in 2_calc_dipole.py (positions centred on the centre of mass, cell zero, N_MAX_D = 6).
        self.d_elems = sorted(["C", "H", "O", "N"]); self.d_map = {s: i for i, s in enumerate(self.d_elems)}
        self.d_desc = jax.jit(PowerSpectrumGenerator(6, dp.r_cut, len(self.d_elems), MAX_NEIGHBORS).process_data)
        self.epnn = EPNN(len(self.d_elems), dp.embed_dim, MLP(dp.update_model_widths),
                         tuple(ResNetCore(dp.message_generator_widths) for _ in range(dp.n_passes)),
                         tuple(ResNetCore(dp.pass_generator_widths) for _ in range(dp.e_passes)))

    def calculator(self, **kw):
        from neuralil.ase_integration_gai import NeuralILASECalculator
        return NeuralILASECalculator(self.ensemble, self.ff, MAX_NEIGHBORS, **kw)

    def dipole(self, atoms):
        import jax.numpy as jnp
        from epnn.preprocessing import create_batched_graph_list
        from epnn.utils import get_nodes_from_graph
        dp = self.dp
        w = np.array([MASS[s] for s in atoms.get_chemical_symbols()])
        p = atoms.positions - (w[:, None] * atoms.positions).sum(0) / w.sum()
        t = jnp.array([self.d_map[s] for s in atoms.get_chemical_symbols()]); c = jnp.zeros((3, 3), jnp.float32)
        pj = jnp.asarray(p, jnp.float32); d = self.d_desc(pj, t, c)
        g = create_batched_graph_list(pj[None], c[None], d.reshape(1, d.shape[0], -1), t[None],
                                      jnp.zeros((1, len(p), 1)), dp.r_cut, dp.r_switch, dp.node_state_dim, 1)
        q = np.asarray(jnp.squeeze(get_nodes_from_graph(self.epnn.apply(dp.params, g[0]))[0]))
        return (q[:, None] * p).sum(0), q
