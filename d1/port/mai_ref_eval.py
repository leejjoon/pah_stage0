# Reference evaluation of the Mai et al. 2025 models in the authors' own pinned environment (Python 3.10,
# JAX/jaxlib 0.4.19 CPU, Flax 0.7.4, NumPy 1.26.1, ASE 3.23.0), UNPATCHED: the models are built as in their
# code/1_MD_calc_position.py, and the dipole follows code/2_calc_dipole.py, whose read_coord() is taken from
# their source file, not copied. Writes <outdir>/ref.npz for mai_compare.py.
# Usage: python mai_ref_eval.py <Supplementary_Data_Code dir> <outdir>      (outdir holds geoms/*.xyz)
import sys, os, ast, glob, pickle, importlib
import numpy as np

root, outdir = sys.argv[1], sys.argv[2]
sys.path[:0] = [f"{root}/lib/bessel-nn-potentials-velo/src", f"{root}/lib/epnn-main/src"]
import jax, jax.numpy as jnp
from ase.io import read
from neuralil.ase_integration_gai import NeuralILASECalculator
from neuralil.bessel_descriptors import PowerSpectrumGenerator
from neuralil.model_older import ResNetCore as ResNetCoreOld, NeuralILwithMorse
from neuralil.model import ResNetCore
from neuralil.plain_ensembles.model import PlainEnsemblewithMorse
from epnn.model import EPNN, MLP
from epnn.preprocessing import create_batched_graph_list
from epnn.utils import get_nodes_from_graph

class Safe(pickle.Unpickler):          # same allow-list as the port; the authors used plain pickle.load
    ALLOWED = {("datetime", "datetime"), ("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
               ("jax._src.array", "_reconstruct_array"),
               ("neuralil.model", "NeuralILModelInfo"), ("epnn.model", "EPNNModelInfo")}
    def find_class(self, mod, name):
        if (mod, name) in self.ALLOWED: return getattr(importlib.import_module(mod), name)
        raise pickle.UnpicklingError(f"blocked global {mod}.{name}")
load = lambda p: Safe(open(p, "rb")).load()

# their read_coord(), executed from their source
src = open(f"{root}/code/2_calc_dipole.py", encoding="utf-8").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "read_coord")
ns = {"jnp": jnp}; exec(compile(ast.Module([fn], []), "2_calc_dipole.py", "exec"), ns); read_coord = ns["read_coord"]

print("jax", jax.__version__, jax.devices())
# energy model: as in 1_MD_calc_position.py
MAX_NEIGHBORS, N_ensembles = 25, 5
model_info = load(f"{root}/model/model_enery.pkl")
descriptor_generator = PowerSpectrumGenerator(model_info.n_max, model_info.r_cut, len(model_info.sorted_elements), MAX_NEIGHBORS)
individual_model = NeuralILwithMorse(len(model_info.sorted_elements), model_info.embed_d, model_info.r_cut,
                                     descriptor_generator.process_data, descriptor_generator.process_some_data,
                                     ResNetCoreOld(model_info.core_widths))
dynamics_model = PlainEnsemblewithMorse(individual_model, N_ensembles)
calculator = NeuralILASECalculator(dynamics_model, model_info, MAX_NEIGHBORS)

# dipole model: as in 2_calc_dipole.py
N_MAX_D, N_BATCH_D, max_neighbors = 6, 1, 25
model_info_dipole = load(f"{root}/model/model_dipole.pkl")
sorted_elements = sorted(['C', 'H', 'O', 'N']); n_types = len(sorted_elements)
get_bessel_descriptor = jax.jit(PowerSpectrumGenerator(N_MAX_D, model_info_dipole.r_cut, n_types, max_neighbors).process_data)
model_dipole = EPNN(n_types, model_info_dipole.embed_dim, MLP(model_info_dipole.update_model_widths),
                    tuple([ResNetCore(model_info_dipole.message_generator_widths) for _ in range(model_info_dipole.n_passes)]),
                    tuple([ResNetCore(model_info_dipole.pass_generator_widths) for _ in range(model_info_dipole.e_passes)]))

names, E, F, Q, MU = [], [], [], [], []
for f in sorted(glob.glob(f"{outdir}/geoms/*.xyz")):
    atoms = read(f); atoms.calc = calculator
    E.append(atoms.get_potential_energy()); F.append(atoms.get_forces())
    cell_list, type_train_list, position_list, init_charges_list = read_coord(f)
    descriptors = jnp.asarray([get_bessel_descriptor(position_list[i], jnp.array(type_train_list[i]), cell_list[i])
                               for i in range(len(type_train_list))])
    descriptors_list = descriptors.reshape(*descriptors.shape[:2], -1)
    graphs = create_batched_graph_list(jnp.array(position_list), jnp.array(cell_list), descriptors_list,
                                       jnp.array(type_train_list), jnp.array(init_charges_list),
                                       model_info_dipole.r_cut, model_info_dipole.r_switch, model_info_dipole.node_state_dim, N_BATCH_D)
    pred = np.asarray(jnp.squeeze(jnp.vstack(list(map(
        lambda g: get_nodes_from_graph(model_dipole.apply(model_info_dipole.params, g))[0], graphs)))))
    q = jnp.expand_dims(jnp.asarray(pred), -1)
    MU.append(np.asarray(sum(q * position_list[0]))); Q.append(pred)
    names.append(os.path.basename(f)[:-4])
    print(f"{names[-1]:24s} E {E[-1]:.4f}  max|F| {np.linalg.norm(F[-1], axis=1).max():.4f}  mu {np.round(MU[-1], 5)}", flush=True)
np.savez(f"{outdir}/ref.npz", names=names, E=np.array(E), **{f"F_{n}": x for n, x in zip(names, F)},
         **{f"Q_{n}": x for n, x in zip(names, Q)}, MU=np.array(MU))
