# Handoff: PySCF and GPU4PySCF on the GH200, and whether a JAX port is feasible (2026-11-13)

For an agent picking this up with no other context. Read this file first, then the files it names. Project: the D1
MLMD work of "Interpretable PAH Diagnostics by Symbolic Regression" (`CLAUDE.md`, `docs/plan/big_picture_proposal.md`).
Repository: this one; branch `charged-mlff` (start your own branch from it, e.g. `pyscf-gpu`).

## 1. Why

We are building DFT training data for machine-learned force fields of PAH neutrals and ions
(`docs/plan/charged_mlff_proposal.md`, `docs/plan/charged_mlff_plan.md`): B3LYP/6-31G\* energies, forces and dipoles on
~7,500 seed geometries, then ~30–50k more by active learning, for PAHs of 10–100 carbon atoms in charge states −1 to
+2. The plan runs Gaussian 16 on KISTI Hangang's x86 CPU nodes (~15k core-h for the seed set, ~100–200k in all).

Two questions for you:
1. Can **GPU4PySCF** on the project's NVIDIA GH200 (aarch64 + H100) produce the same numbers, faster? Then the GH200
   could generate data, and a code we can modify would sit under the whole pipeline.
2. Is a **JAX port** of the part of PySCF we need feasible and worthwhile? Our MD stack is JAX (container A); a JAX
   DFT could be batched over many geometries of one molecule (vmap) and differentiated end to end.

Deliver evidence and a recommendation, not a production system.

## 2. Exactly what the DFT must do

| Requirement | Detail |
| --- | --- |
| Method | B3LYP with VWN3 correlation, i.e. Gaussian's B3LYP (PySCF `xc="b3lypg"`; PySCF's `b3lyp` uses VWN5 and differs) |
| Basis | 6-31G\* with **Cartesian (6D) d functions**, as Gaussian uses by default for 6-31G\* (Gaussian's basis-set page lists the 6-31G family among the Cartesian-d exceptions; checked 2026-11-13). PySCF's default is spherical (5D): set `mol.cart = True`. 4-31G has no d functions, so this does not matter there |
| Wavefunctions | RKS for closed shells; UKS for doublets (cations, anions) and triplets (some dications) |
| Outputs per geometry | Energy; analytic forces; dipole moment (stored about the centre of mass: μ_COM = μ_origin − q R_COM); ⟨S²⟩ before (and ideally after) annihilation; SCF cycle count; convergence flag |
| Stability analysis | Internal instability for UKS (with re-optimisation), and RKS→UKS instability for closed shells (test only); Gaussian's `Stable=Opt` / `Stable` |
| SCF robustness | Distorted, hot geometries of open-shell ions; Gaussian uses `SCF=XQC` (quadratic convergence fallback). PySCF: `mf.newton()` fallback |
| Grid | Gaussian `Int=UltraFine` (99 radial, 590 angular, pruned) on every production job |
| Convergence | Gaussian `SCF=Conver=8` (density ~1e-8) |
| Sizes | 18–200 atoms; 6-31G\* gives ~15 basis functions per C and 2 per H with 6D: ~150 (C10H8) to ~1,600 (C100H30) |
| Elements | C, H now; N and O later |
| Nice to have | Analytic Hessian and IR intensities (to compare with PAHdb harmonic spectra) |

Density fitting: our one test (PySCF CPU, default auxiliary basis, 4-31G) was only 0.9–2.5× faster and moved the
largest force component by up to 2×10⁻³ Ha/bohr, comparable to the force error the models aim for (~1.2×10⁻³
Ha/bohr). The committee review asks for a retest with a proper JK-fitting auxiliary basis before ruling it out.

## 3. What is already established

- **CPU PySCF:** micromamba env `/var/tmp/jjlee-pah/mamba/dft` (Python 3.11, PySCF 2.15.0 from conda-forge, OpenBLAS
  with OpenMP). Micromamba: `/var/tmp/jjlee-pah/mamba/bin/micromamba`, `MAMBA_ROOT_PREFIX=/var/tmp/jjlee-pah/mamba`.
  A separate env `/var/tmp/jjlee-pah/mamba/xtb` has tblite (GFN2-xTB) and ASE.
- **Timing pilot** (`d1/dft_timing.py`; results `/var/tmp/jjlee-pah/d1/dft_pilot/out/*.json`): B3LYP(VWN3)/4-31G,
  energy + forces + dipole, geometry displaced 0.05 Å, 8 Grace threads. Core-hours per point, neutral / cation:
  C10H8 0.013 / 0.018; C24H12 0.089 / 0.15; C32H14 0.29 / 0.47; C54H18 0.90 / 1.30; C90H24 3.2 / 5.3; C144H30 9.4 /
  13.4. Cost ∝ (basis functions)^2.6. Cation ⟨S²⟩ grows with size (0.76 at C10 → 1.04 at C144). Thread scaling: 8
  threads at 84–94 % efficiency, 32 threads 44 % (C24) to 87 % (C144). Table in `docs/plan/charged_tdep_validation.md`.
- **GPU4PySCF:** no aarch64 wheel on PyPI (`pip download gpu4pyscf-cuda12x` found none, 2026-11-12). It would have to
  be built from source.
- **DFT MD running now:** `d1/aimd_pyscf.py`, eight PySCF B3LYP/6-31G\* (6D) trajectories of naphthalene neutral and
  cation, detached, on **CPU cores 0–63**, for several days (`/var/tmp/jjlee-pah/d1/aimd/`). Measured 8.1 s
  (neutral) and 11.6 s (cation) per MD step at 8 threads, with the density reused from the previous step (with 5D d
  functions; the 6D rerun is slightly slower). **Do not stop or disturb these runs, and do not use cores 0–63.**
- **Gaussian Phase 0** (to be run on Hangang by the user; inputs in `/var/tmp/jjlee-pah/d1/g16_phase0/`, reference
  values in its `expected.json`): the pilot geometries at 4-31G and 6-31G\*, a dipole-origin test, two stability jobs,
  and naphthalene at its PAHdb geometries with frequencies. When its logs exist, they are the ground truth for
  matching PySCF and GPU4PySCF to Gaussian.
- **PAHdb reference energies** (B3LYP/6-31G\*, Gaussian, PAHdb v4.00 XML `/var/tmp/jjlee-pah/data/pahdb-complete-theoretical-v4.00.xml`):
  naphthalene neutral uid 330 −385.89270636 Ha, cation uid 359 −385.61361154 Ha, at their own PAHdb geometries
  (`/var/tmp/jjlee-pah/d1/mlff_seed/xyz/C10H8_330_q0.xyz`, `_q1.xyz`).
- **Hardware:** the project node is a dual GH200: 2 × H100 (144 GB HBM3e each; Hangang's are 96 GB), 144 Grace cores, aarch64, 64 KiB
  pages, driver 595.45 (CUDA 13.2). NUMA nodes 0 (cores 0–71) and 1 (72–143) hold the CPUs. Container A
  (`/var/tmp/jjlee-pah/sif/A_jax.sif`, see `containers/A_jax.def`) has JAX 0.11.2 with CUDA 12.9 pip wheels; run it
  with `apptainer exec --nv` and `APPTAINERENV_JAX_PLATFORMS=cuda,cpu`. Apptainer 1.5.5, builds with `--fakeroot`.
  `CLAUDE.md` lists the container rules (one CUDA per image, etc.).
- **GPUs are shared** with other users' jobs: on 2026-11-13 both were at 100 % and ~110 GB. Check `nvidia-smi` before
  any GPU work; use only a GPU that is free, and ask the user if none is.

## 4. Tasks

### T1. PySCF settings that reproduce Gaussian (CPU, no new installs; start here)

1. Naphthalene neutral (RKS) and cation (UKS) at their PAHdb geometries, B3LYP(`b3lypg`)/6-31G\*, `mol.cart=True`,
   tight convergence, a grid as close to Gaussian UltraFine as PySCF allows (`grids.atom_grid = (99, 590)`; try the
   pruning options). Compare with the PAHdb energies above. **Target: |ΔE| ≤ 1×10⁻⁵ Ha**; report how ΔE depends on
   the grid and on 5D vs 6D (expect 5D to be off by ~mHa).
2. Record the PySCF settings that match, and the cost of each setting on CPU (8 threads, cores 72+).
3. When the Phase 0 Gaussian logs arrive (ask the user), compare energies, forces and dipoles on the same geometries.
   Targets: energy ≤ 1×10⁻⁵ Ha, forces ≤ 1×10⁻⁴ Ha/bohr, dipole ≤ 0.01 D.
4. Retest density fitting with a JK-fitting auxiliary basis on the same molecules: speed-up and force error.

### T2. GPU4PySCF on the GH200

1. Survey the build: GPU4PySCF's repository and docs (version, licence, CUDA and cuTENSOR / CuPy requirements,
   aarch64 support, whether `cart=True`, UKS gradients, dipoles and stability analysis are implemented).
2. Build it in a container (new recipe, e.g. `containers/D_gpu4pyscf.def`; do not modify container A). One CUDA in
   the image. Build to `/var/tmp/jjlee-pah/sif/`, never into the repository. Ask the user before downloading large
   images or packages not yet approved.
3. Validate against CPU PySCF (same settings as T1) on naphthalene and coronene, neutral and cation: energy
   ≤ 1×10⁻⁷ Ha, forces ≤ 1×10⁻⁵ Ha/bohr, dipole ≤ 1×10⁻³ D, ⟨S²⟩ ≤ 1×10⁻⁴.
4. Time B3LYP/6-31G\* (6D) energy + forces + dipole on the pilot geometries (C10, C24, C32, C54, C90, C144; geometries
   in `/var/tmp/jjlee-pah/d1/dft_pilot/xyz/`, displaced as in `d1/dft_timing.py`, seed 0). Report seconds per point,
   GPU memory, and the GPU-hour cost of the seed set (7,424 points; size distribution in
   `/var/tmp/jjlee-pah/d1/mlff_seed/species_v2.csv`, 10–60 C) against ~15k CPU core-hours of Gaussian.
5. Check batch throughput: several small molecules at once on one GPU (separate processes or MPS) versus one at a time.

### T3. Feasibility of a JAX port

A desk study first, then a small prototype only if the study says it is worth it.

1. **Survey existing JAX quantum chemistry**, from primary sources (papers, repositories), with versions and dates:
   for example PySCFAD (automatic differentiation for PySCF with a JAX backend), Graphcore's MESS, Grad DFT, and any
   others you find. Do not rely on summaries; mark anything you could not verify.
2. **For each, against §2:** Gaussian-type integrals up to d (Cartesian) with derivatives; two-electron integrals
   (direct, density-fitted, or full tensors: a full ERI tensor for 1,600 basis functions is ~6.6 TB in float64 even
   with 8-fold symmetry, so only direct or density-fitted schemes are viable for large PAHs); B3LYP(VWN3) exchange–correlation on a molecular
   grid; UKS; SCF convergence aids (DIIS, level shift, second-order); stability analysis; forces (analytic or by
   autodiff, and their memory cost); float64 on H100 (FP64 tensor-core throughput is far below FP32).
3. **What a JAX port would buy us**: batching many geometries of one molecule (vmap over a fixed basis layout; our
   seed set is 32 frames per species); end-to-end gradients (e.g. dipole derivatives); one stack with the MD. And
   what it would cost: engineering effort (person-weeks), validation against Gaussian, performance against
   GPU4PySCF's hand-written CUDA kernels.
4. **Recommendation** with an effort estimate: use GPU4PySCF, extend an existing JAX code, write our own, or stay with
   Gaussian on CPU.

## 5. Constraints

- Do not touch the running DFT MD (cores 0–63) or other users' GPU jobs.
- New software goes into new micromamba envs or new containers under `/var/tmp/jjlee-pah/`; do not change the
  existing envs (`dft`, `xtb`) or containers A–C. Ask the user before installing packages from new sources.
- Never print the contents of `.env`.
- Do not send project details, file contents or names to external services; search the web with generic terms.
- Mai et al.'s data and models (Zenodo 15771437, CC BY-NC-SA) stay out of git; two pilot geometries (C90H24_618,
  C144H30_641) come from that release, so do not commit them.
- Commit scripts and the report on your own branch; push only when the user asks.

## 6. Deliverables

- `docs/research/pyscf_gpu_jax.md`: T1–T3 results, with measured numbers marked as measured (with dates) and
  everything else as sourced or estimated; a recommendation and the next step.
- Scripts in `d1/` (e.g. `d1/dft_gpu_timing.py`), a container recipe if you build one, and the settings that match
  Gaussian (they should then be used in `d1/aimd_pyscf.py`, `d1/dft_timing.py` and the QA tools; note it, do not
  change the running MD).

## 7. Files to read

- `docs/plan/charged_mlff_proposal.md` (§2.3 level of theory, §3.4 DFT cost, §5.2 protocol)
- `docs/plan/charged_mlff_plan.md` (run-book; data layout; QA)
- `docs/plan/charged_mlff_committee_review.md` (M2 electronic-state QA, M5 density fitting and ORCA remark)
- `d1/dft_timing.py`, `d1/aimd_pyscf.py`, `d1/g16_io.py` (route lines and what is parsed), `d1/g16_phase0.py`
- `CLAUDE.md` (hardware, containers, security rules)
