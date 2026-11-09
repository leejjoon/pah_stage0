"""Parse `pip show` dumps from the built images (data/pipshow_<image>.txt) into data/deps.json: for each image, every
installed distribution with version, summary, requires and required-by, and whether the recipe asked for it by name.
The dumps were taken with (CPU only, NUMA node 1):
  apptainer exec <image>.sif bash -c 'python -m pip list --format=freeze | sed "s/[=@ ].*//" | xargs python -m pip show \
      | grep -E "^(Name|Version|Summary|Requires|Required-by):"'
(container A and C call /opt/venv/bin/python; B uses the NGC image's own python)."""
import json, re, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()

# Distributions each recipe names explicitly (containers/*.def). B_torchsim adds one to B_torch.
REQUESTED = {
    "A_jax": ["torch", "mace-jax", "jax-md", "flax", "optax", "ase", "numpy", "scipy", "h5py", "matplotlib", "mace-torch"],
    "B_torchsim": ["mace-torch", "cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu13", "ase", "h5py",
                   "torch-sim-atomistic"],
    "C_cpu": ["pysr", "astropy", "specutils", "scikit-learn", "pandas", "h5py", "matplotlib", "jupyter", "pahfit",
              "amespahdbpythonsuite"],
}
# Image B: the 36 distributions our two recipes added on top of NGC 26.10 (their dist-info directories are dated
# 2026-11-07, the build day; the other ~250 are dated 2026-10-09, the NGC release build). Everything else in B is NGC's.
ADDED_B = """ase blosc2 configargparse cuequivariance cuequivariance-ops-cu13 cuequivariance-ops-torch-cu13 cuequivariance-torch
e3nn gitdb gitpython h2 h5py hpack hyperframe lightning-utilities lmdb mace-torch matscipy msgpack ndindex numexpr
nvalchemi-toolkit-ops nvidia-cublas nvidia-cuda-nvrtc opt-einsum opt-einsum-fx orjson prettytable py-cpuinfo smmap
tables threadpoolctl torch-ema torch-sim-atomistic torchmetrics warp-lang""".split()

def parse(path):
    pk, cur = {}, None
    for line in open(path):
        k, _, v = line.rstrip("\n").partition(": ")
        if k == "Name":
            cur = {"name": v, "version": "", "summary": "", "requires": [], "required_by": []}
            pk[norm(v)] = cur
        elif k == "Version": cur["version"] = v
        elif k == "Summary": cur["summary"] = v
        elif k == "Requires": cur["requires"] = [norm(x) for x in v.split(", ") if x]
        elif k == "Required-by": cur["required_by"] = [norm(x) for x in v.split(", ") if x]
    return pk

# Our own one-line role for the packages that matter to the project (shown by the HTML package explorer next to
# the package's own summary). Keyed by normalised name; a dict value gives per-image text.
ROLES = {
    "mace-jax": "MACE in JAX: the model, the weight converter mace-jax-from-torch and the predictor. The JAX side of T0.",
    "jax": "The numerical engine of container A: arrays, autodiff, and the XLA compiler that builds the GPU program.",
    "jaxlib": "JAX's compiled core (XLA runtime).",
    "jax-cuda12-plugin": "JAX's GPU backend (CUDA 12).", "jax-cuda12-pjrt": "JAX's GPU runtime (CUDA 12).",
    "jax-md": "MD in JAX: the nvt_langevin integrator used by t0/md_jax.py.",
    "e3nn-jax": "Equivariant layers (irreducible representations of rotations) in JAX; MACE's building blocks.",
    "jraph": "Padded graph format with node/edge caps; mace-jax batches molecules this way.",
    "flax": "Neural-network modules for JAX; mace-jax's model code.", "optax": "Optimisers for JAX (training; pulled in).",
    "chex": "Test and shape-checking helpers for JAX.",
    "cuequivariance": "NVIDIA's description of equivariant polynomials (pure Python); shared by both stacks.",
    "cuequivariance-jax": "cuEquivariance's JAX front end; with conv_fusion it calls the CUDA kernel for the convolution.",
    "cuequivariance-ops-jax-cu12": "Compiled cuEquivariance kernels for JAX (CUDA 12).",
    "cuequivariance-ops-cu12": "The cuEquivariance CUDA library libcue_ops.so (CUDA 12).",
    "cuequivariance-torch": "cuEquivariance modules for PyTorch; MACE uses them with enable_cueq=True (88 modules).",
    "cuequivariance-ops-torch-cu13": "Compiled cuEquivariance kernels for PyTorch (CUDA 13); does not link libtorch.",
    "cuequivariance-ops-cu13": "The cuEquivariance CUDA library libcue_ops.so (CUDA 13).",
    "torch": {"A_jax": "CPU-only PyTorch: needed to read the PyTorch checkpoint during conversion. Installed FIRST on purpose.",
              "B_torchsim": "NVIDIA's PyTorch build from the NGC image; the constraints file stops pip from replacing it."},
    "e3nn": "Equivariant layers in PyTorch; MACE (PyTorch) is built on it.",
    "mace-torch": {"A_jax": "Reference MACE in PyTorch; here only so the converter can load the checkpoint.",
                   "B_torchsim": "Reference MACE in PyTorch: mace_off() loader, ASE calculator, TorchSim adapter."},
    "ase": "Atomic Simulation Environment: Atoms, units, file formats, Langevin integrator.",
    "matscipy": "Fast neighbour lists for ASE (used by MACE).",
    "torch-sim-atomistic": "TorchSim: batched MD of many systems in one PyTorch state; the batched PyTorch candidate.",
    "nvalchemi-toolkit-ops": "NVIDIA GPU neighbour lists used by TorchSim.",
    "warp-lang": "NVIDIA Warp: a Python-embedded GPU kernel language, used by nvalchemi.",
    "triton": "GPU kernel compiler used inside PyTorch (from NGC).",
    "h5py": "HDF5 files (slice.h5, trajectories).",
    "pysr": "Symbolic regression: evolutionary search over formulas (slice stage 8).",
    "juliacall": "Calls Julia from Python; PySR's bridge to SymbolicRegression.jl.",
    "juliapkg": "Installs Julia and Julia packages; its project dir must be writable at run time.",
    "pahfit": "Decomposes an IR spectrum into continuum, PAH bands and lines (slice stage 6). From git, not PyPI.",
    "amespahdbpythonsuite": "PAHdb tools: database reader, geometry/edge codes, cascade emission model (stages 1-3).",
    "astropy": "Astronomy core: units, tables, FITS, modelling.",
    "specutils": "Spectrum1D objects: PAHFIT's input format.",
    "scikit-learn": "Random-forest reference, isotonic one-ratio baseline (slice stage 7).",
    "pandas": "Tables (features.csv, species.csv).",
    "numpy": "Arrays.", "scipy": "Numerical routines (integration, optimisation, special functions).",
    "pymatgen": "Materials-science toolkit pulled in by jax-md; not used by T0.",
}

out = {}
for img, req in REQUESTED.items():
    pk = parse(os.path.join(DATA, f"pipshow_{img}.txt"))
    for n in map(norm, req):
        assert n in pk, (img, n)
        pk[n]["requested"] = True
    if img == "B_torchsim":
        for n, p in pk.items(): p["origin"] = "added" if n in ADDED_B else "ngc"
    for n, r in ROLES.items():
        if n in pk: pk[n]["role"] = r.get(img, "") if isinstance(r, dict) else r
    out[img] = pk
    print(f"{img}: {len(pk)} distributions, {sum(1 for p in pk.values() if not p['required_by'])} with no dependants")
json.dump(out, open(os.path.join(DATA, "deps.json"), "w"), indent=0)
