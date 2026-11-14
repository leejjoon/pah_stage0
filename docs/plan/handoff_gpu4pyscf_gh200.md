# Handoff: GPU4PySCF on the GH200, then the seed-set DFT (2026-11-13)

For an agent picking this up on the project GH200 (`gh200`) with no other context. Read this file, then
`docs/research/pyscf_gpu_jax.md` (the results this builds on) and `CLAUDE.md` (GH200 rules: aarch64, Apptainer,
one CUDA per image, MPS). The earlier handoff, `docs/plan/handoff_pyscf_gpu_jax.md`, describes the original task;
its §2 (what the DFT must do) still holds. Branch: `feat-gpu` (commits `64b3aba`, `56c76ff`, `a87983a`).

## 1. Where things stand

The previous study ran on `wasp33a` (x86, 4× RTX 5000 Ada, FP64 ≈ 1/64 FP32), not the GH200. It concluded:

- **Settings that match Gaussian** (PAHdb naphthalene energies to ≤ 2 µHa, IR intensities to ~2 %):
  ```python
  mol = gto.M(atom=..., basis="6-31g*", cart=True, charge=q, spin=mult - 1)    # 6D d, as Gaussian
  mf = (dft.RKS if mult == 1 else dft.UKS)(mol); mf.xc = "b3lypg"
  mf.grids.atom_grid = (99, 590)
  mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200
  mf = mf.to_gpu()                                                               # GPU4PySCF
  # fallback: if not mf.converged: mf = mf.newton(); mf.kernel(mf.mo_coeff, mf.mo_occ)
  ```
  Keep `conv_tol_grad`: without it cation dipoles drift by up to 1e-3 D. Do not tighten `conv_tol` below ~1e-9 on
  the GPU: for ≥ ~850 basis functions the GPU SCF energy has a ~1e-9 Ha noise floor and never converges.
- **GPU4PySCF 1.9.2 = CPU PySCF** (ΔE ≤ 2e-10 Ha, forces ≤ 7e-7 Ha/bohr, dipole ≤ 1e-4 D, ⟨S²⟩ ≤ 3e-7), UKS,
  gradients, dipoles, ⟨S²⟩ and stability analysis included.
- **Timings on one RTX 5000 Ada** (energy + forces + dipole, s per point, q0 / q1): C10 12 / 19, C24 37 / 74, C32
  58 / 106, C54 150 / 229, C90 452 / 793, C144 1123 / 1733. Seed set (7,424 points) ≈ 280 GPU-hours.
- **Several processes per GPU (no MPS) gave no throughput gain** (ovalene: 59 → 56 points/h with 4 processes).
  Untested: concurrent execution (MPS) and whether the FP64 units are actually saturated.
- **JAX port: not worth it now** (survey in the report, §3). Use GPU4PySCF.
- **Gaussian is not available on Hangang**; PAHdb is the only Gaussian reference. Phase 0 will not run.

## 2. First: check the running DFT MD for the Newton-fallback bug

The DFT MD started from `d1/aimd_pyscf.py` (eight naphthalene q0/q1 trajectories, `/var/tmp/jjlee-pah/d1/aimd/`,
cores 0–63) runs the code *before* commit `64b3aba`. Its fallback was `mf.newton(); mf.kernel(mf.make_rdm1())`,
which passes a density where `newton().kernel` expects MO coefficients and returns a non-physical energy. Any MD
step that needed more than 200 SCF cycles is wrong, and the trajectory continued from it.

- Check, read-only: the logs and the energy series in the outputs for a jump (hundreds to thousands of Ha) or a
  Newton/unconverged message. Do **not** stop, restart or touch the processes; report to the user.
- New runs use the fixed script; also switch them to the (99, 590) grid (the script still uses PySCF's default
  `level=3`, ~45 µHa off from C24 up; fine for C10).

## 3. Tasks

### G1. Build GPU4PySCF for aarch64 + H100

There is no aarch64 wheel of `gpu4pyscf-cuda12x` on PyPI (checked 2026-11-12). Build from source, pinned to the
versions validated on wasp33a:

| Package | Version on wasp33a | Notes for aarch64 (unverified unless stated) |
|---|---|---|
| gpu4pyscf | 1.9.2 | Build from the git tag; CMake + nvcc, set `CUDA_ARCHITECTURES=90` (H100) |
| gpu4pyscf-libxc | 0.9.0 | Separate package; check for an aarch64 wheel, else build |
| cupy-cuda12x | 14.2.0 | CuPy publishes aarch64 wheels for some versions; check |
| cutensor-cu12 | 2.8.1 | Optional (GPU4PySCF falls back to CuPy's contraction engine) |
| pyscf | 2.15.0 | conda-forge has aarch64 builds |
| python | 3.11 | |

- Put it in a new container (e.g. `containers/D_gpu4pyscf.def`, built with `apptainer build --fakeroot` to
  `/var/tmp/jjlee-pah/sif/`) or a new micromamba env under `/var/tmp/jjlee-pah/mamba/`. Do not modify containers
  A–C or the envs `dft`, `xtb`. One CUDA in the image. Ask the user before pulling large base images or packages
  from new sources.
- Record the exact build steps in the recipe and in a short `containers/D_gpu4pyscf.md`.

### G2. Validate on the H100 (reuse the scripts)

```bash
# same as wasp33a; geometry first, then output dir
python d1/pyscf_match.py <geom> <outdir> --tag coro_q1 --charge 1 --mult 2 --displace 0.05 --seed 0 \
       --only 99x590_t9g6 --gpu
python d1/pyscf_match_report.py <outdir>
python d1/stability_check.py <geom> --charge 1 --mult 2 --gpu --out r.json
```

- Against CPU PySCF in the `dft` env (cores 72+, never 0–63): naphthalene and coronene, q0 and q1, displaced. Targets
  as before: ΔE ≤ 1e-7 Ha, forces ≤ 1e-5 Ha/bohr, dipole ≤ 1e-3 D, ⟨S²⟩ ≤ 1e-4. Check the noise floor again on
  the H100 (does `conv_tol` 1e-11 converge for circumcoronene?).
- wasp33a reference energies (displaced 0.05 Å, seed 0, t9g6) to compare against directly, in Ha:
  naphthalene (PAHdb geometry `g16_phase0/pahdb/naph_uid330_q0_freq.gjf` / `uid359_q1`) −385.847755548 /
  −385.567354806; pilot `dft_pilot/xyz/`: coronene −921.731205985 / −921.459938733, ovalene −1227.818570140 /
  −1227.596153211, circumcoronene −2068.574980092 / −2068.423402081, C90H24_618 −3443.951432147 / −3443.773599127,
  C144H30_641 −5505.570393449 / −5505.399183752. Agreement should be ~1e-9 Ha.

### G3. Timings and throughput on the H100

- The C10–C144 series (q0, q1) with `pyscf_match.py --gpu --only 99x590_t9g6`, then recompute the seed-set cost
  (7,424 points, `d1/mlff_seed/species_v2.csv`, 124 pairs, 10–96 C; ~30 points per species and charge; power-law
  fit of time against nC).
- T2.5 on the H100 with `d1/dft_gpu_batch.py` (ovalene, 8 frames per process): 1, 2, 4 processes **without** MPS,
  then **with** MPS, run as `CLAUDE.md` describes (short `CUDA_MPS_PIPE_DIRECTORY`; daemon inside the image if the
  image's libcuda differs from the host's). MPS changes the shared GPU's mode: ask the user first, and only on a
  GPU with no other users' jobs. Include naphthalene and coronene: batching is most likely to help small molecules.
- Optional: `nsys profile` of one naphthalene and one ovalene point (kernel sizes, idle gaps) to judge whether
  batching or a JAX vmap could help. `ncu` counters may need admin rights.

### G4. Production driver for the seed set

A `d1/` script that works through `species_v2.csv` and the seed frames (`d1/mlff_seed/xyz/`):

- one process per GPU (or several under MPS, if G3 shows a gain); resumable (skip finished frames); one output file
  per frame or per species;
- per frame: energy, forces, dipole about the centre of mass (μ_COM = μ_origin − q R_COM), ⟨S²⟩, SCF cycles,
  convergence and Newton flags, wall time, versions and host;
- stability analysis (`stability_check.py` logic; UKS internal with re-optimisation, RKS→UKS test) on a sample, not
  every frame: it costs 2–6× an SCF on the GPU;
- QA: flag unconverged frames, Newton fallbacks, ⟨S²⟩ far above 0.75 for doublets, and force outliers;
- a pilot on a few species before the full set; ask the user before starting the full run (GPU time on a shared
  node).

### G5. Documentation

- Add the H100 numbers to `docs/research/pyscf_gpu_jax.md` (measured, dated, which GPU).
- Update `docs/plan/charged_mlff_plan.md` (and wherever the plan assumes Gaussian on Hangang) to GPU4PySCF.
- `d1/g16_phase0.py`'s `expected.json` labels PAHdb frequencies "unscaled"; PAHdb v4 stores them scaled, with a
  per-mode `scale` attribute (0.9794 / 0.9691 / 0.9597). Fix the label.

## 4. Data

- On the GH200 already: `/var/tmp/jjlee-pah/d1/{mlff_seed,dft_pilot,g16_phase0,aimd}`, the PAHdb XML.
- On wasp33a only (copy if needed; the agent cannot SSH, the user copies): result JSONs under
  `/var/tmp/jjlee-pah/d1/{t1_match,t1_freq,t2_timing,t2_timing_cpu,t25_batch,stab}`. The report quotes everything
  that matters from them.

## 5. Constraints

- Do not touch the running DFT MD (cores 0–63) or other users' GPU jobs. Check `nvidia-smi` before any GPU work and
  use only a free GPU; ask the user if none is.
- New software only in new containers or envs under `/var/tmp/jjlee-pah/`; ask before installing from new sources.
- Never print `.env`. Do not send project details, file contents or names to external services; search the web with
  generic terms.
- C90H24_618 and C144H30_641 come from Mai et al. (Zenodo 15771437, CC BY-NC-SA): never commit them or results
  files that contain their geometries.
- Commit on your own branch (start from `feat-gpu`); push only when the user asks.
