# PySCF, GPU4PySCF and JAX for the B3LYP/6-31G\* training data (2026-11-13)

Answers `docs/plan/handoff_pyscf_gpu_jax.md` (T1–T3). **Measured** numbers were measured by us on 2026-11-12/13;
everything else is marked *sourced* (with the source) or *estimated*.

## 0. Summary and recommendation

- **PySCF reproduces Gaussian's B3LYP/6-31G\*** to ≤ 2 µHa against PAHdb's Gaussian energies (target 10 µHa) with
  `xc="b3lypg"`, `cart=True` (6D d) and `grids.atom_grid = (99, 590)`. 5D d functions are off by 8 mHa; PySCF's
  default grid by ~10–45 µHa. IR intensities from PySCF match PAHdb's to ~2 % (strong bands). *Measured.*
- **GPU4PySCF 1.9.2 matches CPU PySCF** (ΔE ≤ 2e-10 Ha, forces ≤ 7e-7 Ha/bohr, dipole ≤ 1e-4 D, ⟨S²⟩ ≤ 3e-7),
  including UKS, forces, dipoles and stability analysis, and on a single RTX 5000 Ada it is 4–7× faster than 16 CPU
  threads for C24–C54. *Measured.*
- **Seed set (7,424 points): ≈ 280 GPU-hours on one RTX 5000 Ada** (≈ 3 days; < 1 day on 4). *Estimated from
  measured timings.* This card has weak FP64 (1/64 of FP32); an H100 should be several times faster (*not measured*).
- **Several processes on one GPU do not raise throughput** without MPS: one process keeps the GPU busy almost all
  the time, so time-slicing has no idle gaps to fill. Whether *concurrent* kernels (MPS, or a JAX vmap batch) would
  help is not measured: "busy" is not the same as FP64-saturated. For now scale by GPUs. *Measured, no MPS.*
- **A JAX port is not worth it now.** No existing JAX code covers §2 of the handoff at PAH sizes; the closest
  (dftax) lacks stability analysis and ⟨S²⟩, batches only up to ~70 basis functions and is ~30× slower than
  GPU4PySCF. Extending it: 10–20 person-weeks; a new code: 40–80. *Sourced desk study, §3.*
- **Recommendation: use GPU4PySCF** (1–3 person-weeks of wiring and QA) with the settings in §1.3, one process per
  GPU, many GPUs. Keep Gaussian only as an occasional spot check if access ever appears.
- **Bug found and fixed** (commit `64b3aba`): the Newton fallback in `d1/aimd_pyscf.py` and `d1/dft_timing.py`
  passed a density matrix where `newton().kernel` expects MO coefficients, giving nonsense energies (−3492 Ha for
  circumcoronene, true −2068.57). **The DFT MD running on the GH200 uses the old code**: any step that needed more
  than 200 SCF cycles is wrong. None of the 25 pilot runs needed the fallback.

**Update 2026-11-14 (GH200, §5):** GPU4PySCF now runs on the GH200's H100 (container D, built from source). Same
results as CPU PySCF and wasp33a (energies to 1e-9 Ha); 1.1× (C10) to 9× (C144) faster than the RTX 5000 Ada, because
a single SCF keeps the H100 busy only 13–37 % of the time (launch-bound small kernels). Two processes per GPU give
~1.35× from C24 up; MPS adds nothing. Seed set: **≈ 87 H100-hours including stability tests, ≈ 33 h on the node's two
GPUs.** The DFT MD never hit the Newton-fallback bug.

## Setup (deviations from the handoff)

- **Node:** `wasp33a`, not the GH200: 2× Xeon Gold 6526Y (32 cores), 4× RTX 5000 Ada (32 GB, FP64 ≈ 1/64 FP32),
  driver 580.82. Other jobs shared the CPUs during the CPU timings, so CPU times are pessimistic and noisy. GPUs 0
  and 2 were in use by others; our runs used GPU 3 and GPU 1, one at a time per measurement.
- **Software:** PySCF 2.15.0 (conda-forge, OpenMP OpenBLAS), env `~/miniforge3/envs/pyscf`; GPU4PySCF
  `gpu4pyscf-cuda12x` 1.9.2 (Apache-2.0, x86-64 wheel from PyPI), CuPy 14.2.0, cuTENSOR 2.8.1, CUDA 12.9, env
  `~/miniforge3/envs/gpu4pyscf`. No container was built (no apptainer on this node; no aarch64 wheel exists for the
  GH200, see the handoff §3).
- **No Gaussian logs:** Gaussian is not available on Hangang, so Phase 0 (T1.3) will not run. PAHdb v4.00's
  Gaussian energies, frequencies and intensities for naphthalene (uid 330 neutral, 359 cation) are the only
  Gaussian reference; the frequency/IR test (§1.2) stands in for the force and dipole comparison.
- **Scripts** (branch `feat-gpu`): `d1/pyscf_match.py` (one single point per named setting, CPU or `--gpu`),
  `d1/pyscf_match_report.py`, `d1/pyscf_freq_fd.py` (finite-difference frequencies and IR intensities against
  PAHdb), `d1/dft_gpu_batch.py` (T2.5), `d1/stability_check.py`. Outputs under `/var/tmp/jjlee-pah/d1/`
  (`t1_match/`, `t1_freq/`, `t2_timing/`, `t2_timing_cpu/`, `t25_batch/`, `stab/`).

## 1. T1: PySCF settings that reproduce Gaussian

### 1.1 Energies against PAHdb (naphthalene at the PAHdb geometries, 8-decimal coordinates)

All 6D, `b3lypg`, `conv_tol` 1e-11. ΔE = PySCF − PAHdb (uid 330: −385.89270636 Ha; uid 359: −385.61361154 Ha).
*Measured.*

| Setting | ΔE neutral (µHa) | ΔE cation (µHa) |
|---|---|---|
| (99, 590), PySCF default pruning (NWChem) | −0.2 | −0.1 |
| (99, 590), no pruning | −0.1 | 0.0 |
| (99, 590), Stratmann–Scuseria–Frisch weights | −0.7 | −0.8 |
| (99, 590), Murray–Handy–Laming radial + SSF (closest to Gaussian's construction) | +0.2 | +1.7 |
| (200, 974), unpruned (grid-converged reference) | −1.6 | −1.8 |
| (75, 302) and PySCF default `level=3` | −9.3 | −11.6 |
| (99, 590), SG-1 pruning | +33 | +22 |
| **5D d functions** (PySCF default), (99, 590) | **+8050** | **+7947** |
| Density fitting, `def2-universal-jkfit` | −47 | −48 |
| Density fitting, PySCF default auxiliary | +62 | +55 |

PAHdb's route line (`#b3lyp/6-31G* freq ... scf=(xqc,...)`) has no `Int=` keyword, so these used the default grid
of whichever Gaussian version ran them (FineGrid before G16, UltraFine in G16). The energies agree best with the
UltraFine-sized grid; FineGrid-sized grids are ~10 µHa off. Everything within ±2 µHa is at the level of grid
differences.

On displaced geometries (0.05 Å Gaussian displacement, seed 0) of naphthalene and coronene, neutral and cation,
against the (200, 974) reference (*measured*):

| Setting | |ΔE| (µHa) | max |ΔF| (Ha/bohr) | |Δμ| (D) | time vs (99, 590) |
|---|---|---|---|---|
| (99, 590), `conv_tol` 1e-11 | ≤ 1.1 | ≤ 4.5e-6 | ≤ 1.5e-4 | 1 |
| (99, 590), `conv_tol` 1e-9, `conv_tol_grad` 1e-6 (production) | ≤ 1.1 | ≤ 4.5e-6 | ≤ 1.0e-4 | — |
| (99, 590), `conv_tol` 1e-9 only | ≤ 1.1 | ≤ 1.2e-5 | **≤ 1.0e-3** | ~0.9 |
| (75, 302) / `level=3` | up to 44 (coronene) | ~2.7e-5 | — | ~0.6 |
| Density fitting (JK aux) | 50–173 | 1–2.2e-5 | — | 0.3–0.45 (coronene) |

- The gradient criterion matters: without `conv_tol_grad`, cation dipoles drift by up to 1e-3 D.
- Density fitting with a proper JK basis is 2.2–3× faster on CPU for coronene and moves forces by ≤ 2.2e-5
  Ha/bohr (the 4-31G test in the handoff saw 2e-3 with the default auxiliary basis), but it shifts absolute
  energies by 50–170 µHa, beyond the 10 µHa target. Not recommended while Gaussian is the reference.
- **`B3LYP` in PySCF 2.15 is already Gaussian's variant:** `B3LYP` and `B3LYPG` both map to libxc 402 (VWN-RPA,
  Gaussian's "VWN3"); `B3LYP5` is the VWN5 one (*sourced*: `pyscf/dft/libxc.py`, and checked locally). The
  handoff's §2 statement that `b3lyp` uses VWN5 is out of date; `b3lypg` is still the unambiguous spelling.

### 1.2 Frequencies and IR intensities against PAHdb (stand-in for the Phase 0 force/dipole test)

Central finite differences of PySCF analytic gradients and dipoles (step 0.005 bohr), at the PAHdb geometries,
(99, 590), 1e-11, most-abundant-isotope masses (`d1/pyscf_freq_fd.py`; `t1_freq/naph_q{0,1}/compare.json`).
*Measured.*

| | Neutral (uid 330) | Cation (uid 359) |
|---|---|---|
| max force at the PAHdb geometry (Ha/bohr) | 1.4e-4 | 1.5e-4 |
| strong bands (> 1 km/mol): mean / max relative intensity difference | 2.1 % / 5.5 % | 2.2 % / 10.6 % |
| total intensity, PySCF vs PAHdb (km/mol) | 290 vs 286 | 541 vs 539 |
| frequency offset < 1111 / 1111–2500 / C–H stretch (cm⁻¹) | +1.2 / +2.2 / +11.4 | +1.2 / +2.2 / +11.5 |

- PAHdb v4 stores **scaled** frequencies with a per-mode `scale` attribute (0.9794, 0.9691, 0.9597 by range;
  Bauschlicher et al. 2018, ApJS 234, 32, §3 and Table 2). We divided each by its own scale. `expected.json`
  from `d1/g16_phase0.py` calls them "unscaled"; that label is wrong.
- The residual offsets (0.1–0.36 %, uniform within each range) are unexplained. They are not grid (75,302 and
  SG-1 give the same), not finite-difference error (the analytic Hessian agrees), and 5D makes them worse. A
  geometry rounding or a scale factor stored with more digits than the attribute shows are candidates; they do not
  affect energies, forces or dipoles at a given geometry, which is what the training data need.
- The forces at the PAHdb geometries (1.4e-4 Ha/bohr) are what one expects from a Gaussian optimisation to its
  default criteria (max force 4.5e-4), i.e. consistent with the same potential energy surface.

### 1.3 Settings to use

```python
mol = gto.M(atom=..., basis="6-31g*", cart=True, charge=q, spin=mult - 1)      # 6D d, as Gaussian
mf = (dft.RKS if mult == 1 else dft.UKS)(mol)
mf.xc = "b3lypg"                       # VWN3 (libxc 402); "b3lyp" is the same in PySCF >= 2.x
mf.grids.atom_grid = (99, 590)         # UltraFine size; default NWChem pruning
mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200
# GPU: mf = mf.to_gpu()
# fallback: if not mf.converged: mf = mf.newton(); mf.kernel(mf.mo_coeff, mf.mo_occ)
```

`d1/aimd_pyscf.py` and `d1/dft_timing.py` still use PySCF's default grid (`level=3`): fine for C10 (−9 µHa), off by
~45 µHa from C24. Not changed, because the MD is running; switch to (99, 590) for new runs.

## 2. T2: GPU4PySCF

### 2.1 Survey (T2.1)

GPU4PySCF 1.9.2 (2026-11-13; Apache-2.0; *sourced*: PyPI metadata). Wheels for x86-64 only; the GH200 needs a source
build. Needs CuPy ≥ 13 and its own `gpu4pyscf-libxc` build; cuTENSOR optional. Implemented and used here:
`cart=True`, RKS/UKS energies, analytic gradients, dipoles, ⟨S²⟩, Newton (second-order) SCF, RKS internal/external
and UKS internal stability analysis (`gpu4pyscf/scf/stability.py`), all through `mf.to_gpu()`.

### 2.2 Agreement with CPU PySCF (T2.3)

Displaced naphthalene and coronene, neutral and cation, identical settings, conv_tol 1e-11 (*measured*): max ΔE
1.7e-10 Ha, forces 7e-7 Ha/bohr, dipole 9.5e-5 D, ⟨S²⟩ 3e-7. All targets met (1e-7, 1e-5, 1e-3, 1e-4). With the
production tolerance, CPU and GPU energies agree to ~1e-9 Ha up to C54.

**GPU noise floor.** For ≥ ~850 basis functions the GPU SCF energy fluctuates at ~1e-9 Ha (|g| ~5e-8), so
`conv_tol` 1e-11 never converges (circumcoronene ran 200 cycles; then the buggy Newton fallback gave −3492 Ha).
Hence `conv_tol` 1e-9 with `conv_tol_grad` 1e-6, which loses nothing measurable against 1e-11 (§1.1).

### 2.3 Timings (T2.4)

Energy + analytic forces + dipole + ⟨S²⟩ per point; pilot geometries displaced 0.05 Å (seed 0); production
tolerance; GPU after a warm-up; one RTX 5000 Ada (GPU 3); 4 CPU threads for the host side. *Measured.*

| Molecule | nao (6D) | q0: s (cycles) | q1: s (cycles) | GPU memory q0 / q1 (GB) | CPU 16 threads q0 / q1 (s) |
|---|---|---|---|---|---|
| C10H8 | 166 | 12 (13) | 19 (21) | 0.28 / 0.45 | — |
| C24H12 coronene | 384 | 37 (15) | 74 (28) | 0.45 / 0.73 | — |
| C32H14 ovalene | 508 | 58 (15) | 106 (27) | 0.55 / 0.98 | 366 / 457 |
| C54H18 circumcoronene | 846 | 150 (18) | 229 (28) | 0.90 / 1.06 | 877 / 1506 |
| C90H24 | 1398 | 452 (23) | 793 (46) | 1.10 / 1.90 | — |
| C144H30 | 2220 | 1123 (25) | 1733 (46) | 1.69 / 2.93 | — |

- GPU over CPU: 1.5–1.8× (C10) and 4.5–5.3× (C24) over 8 threads; 4.3–6.6× (C32, C54) over 16 threads. The CPU
  cores were shared with other jobs, so the true CPU times are somewhat lower.
- GPU time grows as nC^1.8–1.9 from C24 to C144 (q0/q1). Forces are 10–15 % of the time. Cations take 1.5–2× the
  cycles of neutrals. Cation ⟨S²⟩ rises with size: 0.765 (C10), 0.767 (C24), 0.776 (C32), 0.778 (C54), 0.813 (C90),
  1.040 (C144).
- Memory is small (≤ 3 GB at C144): many processes would fit, but see §2.4.

**Seed-set cost** (7,424 points over the 124 pairs of `species_v2.csv`, 10–96 C, median 36; even split over
species and q0/q1; power-law fit of the timings above): **≈ 280 GPU-hours** on one RTX 5000 Ada, one point at a
time. *Estimated.* At the measured GPU/CPU ratio this corresponds to roughly 20–30k CPU core-hours of PySCF
(*estimated*), against the plan's ~15k core-hours of Gaussian on Hangang. An H100 has ~30–60× this card's FP64 peak
(*sourced*: vendor specs, H100 SXM 34 TFLOP/s vector, 67 with tensor cores; RTX 5000 Ada ~1 TFLOP/s); GPU4PySCF is not purely FP64-bound, so the real gain is smaller and
must be measured.

### 2.4 Several processes on one GPU (T2.5)

Ovalene, 8 displaced frames (0.05 Å) per process, separate processes on GPU 1, no MPS, 4 CPU threads each.
*Measured.*

| Run | Points per hour (total) | s per point per process |
|---|---|---|
| q0, 1 process | 59.3 | 61 |
| q0, 1 process, each SCF started from the previous frame's density | 55.7 | 65 |
| q0, 2 processes | 56.5 | 127 |
| q0, 4 processes | 55.8 | 258 |
| q1, 1 process | 36.5 | 99 |
| q1, 4 processes | 34.4 | 419 |

- Without MPS, kernels from different processes do not run concurrently; the GPU time-slices between contexts.
  Throughput stayed flat because one process leaves the GPU almost never idle (the host-side work between kernels is
  short), so there are no gaps for a second process to fill.
- This does **not** show that the FP64 units are saturated. Kernels can keep the GPU "busy" while using only part
  of it (too few blocks, latency-bound small matrices). Concurrent execution, via MPS or one process running several
  molecules, could still help, most likely for the smaller molecules. Settling it needs either an MPS run (not done:
  the GPUs are shared) or a profile of achieved FP64 throughput per kernel (Nsight Compute).
- Reusing the density does not help for independent random displacements (it saves cycles only along a
  trajectory, as in the MD).
- On a GPU with strong FP64 (H100) with small molecules the balance may differ; recheck there.

### 2.5 Stability analysis

`d1/stability_check.py`: RKS internal + RKS→UKS (Gaussian `Stable`), UKS internal with re-optimisation (Gaussian
`Stable=Opt`), displaced geometries. *Measured.*

| Molecule | GPU: stable? | GPU time: SCF / stability (s) | CPU 8 threads: stable? | CPU time: SCF / stability (s) |
|---|---|---|---|---|
| C10H8 q0 (int + ext) | yes, yes | 12 / 51 | yes, yes | 11 / 156 |
| C10H8 q1 (UKS int) | yes | 19 / 33 | yes | 26 / 124 |
| C24H12 q0 (int + ext) | yes, yes | 33 / 185 | yes, yes | 123 / 2990 |
| C24H12 q1 (UKS int) | yes | 66 / 130 | yes | 368 / 2001 |
| C54H18 q1 (UKS int) | yes | 202 / 591 | — | |

- It works on the GPU and agrees with CPU (same energies to 1e-10 Ha, same verdicts). None of these geometries was
  unstable, so the re-optimisation branch is untested.
- It costs 2–6× an SCF on the GPU (5–24× on CPU, where it is 4–16× slower than on the GPU). Running it on every point would triple the cost; run it on a sample (e.g. the first and a
  few hot frames per species and charge) and on any point whose ⟨S²⟩ or SCF behaviour is unusual.

## 3. T3: Feasibility of a JAX port (desk study)

A literature and repository survey (repositories cloned and read at HEAD on 2026-11-13; *sourced*, unverified items
marked). Details are summarised; the code references are in the survey notes.

| Code (version, date, licence) | Status | Fit to handoff §2 |
|---|---|---|
| **PySCFAD** (fishjojo/pyscfad v0.3.4, 2026-09; Apache-2.0) | Active, one main developer; JCP 157, 204801 (2022) | Integrals and XC on the CPU via libcint/libxc; full ERI tensor or DF, no direct J/K; UKS and ⟨S²⟩ yes; batched jittable SCF is RHF-only with integrals evaluated serially on the CPU |
| **dftax** (andresguzco/dftax v0.9.3, 2026-10; Apache-2.0) | 4 months old, single author | Pure JAX on GPU, Cartesian default, B3LYP with VWN-RPA, UKS, DF, analytic forces, Newton; vmap over geometries only on the exact-ERI path (nao ≲ 70 on an 80 GB A100); no stability, no ⟨S²⟩; coronene/def2-SVP/PBE ~30× slower than GPU4PySCF warm, 100–300 s compile; validated only on water/STO-3G |
| **GradSCF** (STOKES-DOT v0.1.0, 2026-10; MIT) | 2.5 months old, active | libcint on the CPU; direct J/K and RI on CPU; B3LYP-VWN-RPA; UKS; **has** stability analysis (not with DF); slower than PySCF on CPU for 880 AO |
| **MESS** (Graphcore 0.1.2, 2024; MIT) | Dormant | Dense ERI, restricted only, no DIIS; toy molecules |
| **Grad DFT** (Xanadu, 2024; Apache-2.0) | Dormant | No nuclear gradients; B3LYP uses VWN5 |
| **D4FT** (sail-sg, 2024; Apache-2.0) | Dormant | In-core ERI, no hybrids found, toy sizes |

What a JAX port would buy: vmap over the ~32–60 frames per species (fixed basis layout), end-to-end
differentiation (dipole derivatives, response), one stack with the MD. What it costs (*estimated*, person-weeks):

| Path | Effort | Main risk |
|---|---|---|
| Use GPU4PySCF | 1–3 | No vmap (batch with processes/GPUs); no end-to-end autodiff |
| Extend dftax | 10–20 (stability + ⟨S²⟩ 2; batched DF 3–5; XC screening and DF blocking 4–8; analytic dipole derivatives 1–2; GH200 build and validation 2) | May never reach GPU4PySCF's speed; maintaining a fork of a single-author code |
| Write a new JAX code | 40–80 | Matching hand-written CUDA kernels from XLA |

T2.5 does not settle whether batching would pay. A vmap batch runs in one CUDA context, so it is the
concurrent case that MPS tests, not the time-sliced one we measured. It can add throughput only if one molecule's
kernels leave FP64 units idle; it cannot raise the FP64 peak, and on this card (FP64 ≈ 1/64 FP32) mid-sized SCFs are
likely near it. It would also have to overcome dftax's ~2× per-iteration and ~30× end-to-end deficit before it gained
anything. Batching helps only where a single SCF leaves FP64 units idle; that is unmeasured, and most plausible for
the smallest species (C10–C24).

**Recommendation:** use GPU4PySCF. Revisit dftax only if end-to-end differentiation through the SCF becomes a real
requirement.

## 4. Next steps (as of 2026-11-13; see §5 for what was done on the GH200)

1. ~~Check the GH200 DFT MD for the Newton-fallback bug.~~ Done 2026-11-14 (§5.1): never triggered.
2. ~~Production driver for the seed set.~~ `d1/dft_seed_run.py` (§5.5).
3. ~~Measure on an H100.~~ §5.3–5.4.
4. Optional: one Gaussian spot check (energy, forces, dipole on two displaced ions) if Gaussian becomes available
   anywhere, to close T1.3 directly.

## 5. GH200 (H100) results (2026-11-14)

Project node `gh200`: GH200 144 GB (H100, sm_90), driver 595.45 (CUDA 13.2), Grace CPU. Container D
(`containers/D_gpu4pyscf.md`): GPU4PySCF 1.9.2 and gpu4pyscf-libxc 0.9.0 built from source for aarch64/sm_90, CuPy
14.2.0 rebuilt with cuTENSOR (the aarch64 wheel lacks it), PySCF 2.15.0, CUDA 12.9. Runs and tables:
`d1/h100_bench.sh`; outputs under `/var/tmp/jjlee-pah/d1/h100_*`. Everything here is *measured* unless marked.

### 5.1 The GH200 DFT MD and the Newton-fallback bug

The MD (`/var/tmp/jjlee-pah/d1/aimd/`, 8 naphthalene runs) was stopped on 2026-11-13, not running. Its checkpoints
show at most 9 SCF cycles (neutral) and 14 (cation) in the recorded steps (every 20th), no energy jumps (largest
change of the conserved energy between records 0.1–0.3 eV, thermal), dipoles continuous (≤ 0.04 D per step) and
⟨S²⟩ ≤ 0.772. The 200-cycle fallback was never reached, so the 3.0 ps (neutral) and 2.0 ps (cation) done so far are
valid. It still has to be finished on another server (fixed script; the (99, 590) grid is optional for C10).

### 5.2 Agreement on the H100

Same four displaced points as §2.2 (0.05 Å, seed 0, production setting), GPU (container D) against CPU PySCF 2.15.0
(env `dft`, 18 Grace threads):

| Point | ΔE (Ha) | max \|ΔF\| (Ha/bohr) | \|Δμ\| (D) | Δ⟨S²⟩ | cycles CPU / GPU |
|---|---|---|---|---|---|
| C10H8 q0 | 2e-11 | 2.3e-7 | 1.6e-5 | 0 | 13 / 13 |
| C10H8 q1 | 1e-11 | 4.0e-7 | 1.9e-5 | 1e-7 | 20 / 21 |
| C24H12 q0 | 9e-11 | 2.1e-7 | 8e-6 | 0 | 15 / 15 |
| C24H12 q1 | 1.7e-10 | 6.9e-7 | 3.0e-5 | 1.4e-7 | 27 / 28 |

All far inside the targets (1e-7, 1e-5, 1e-3, 1e-4). Every energy of the C10–C144 series (12 points) and the CPU
energies on the Grace CPU equal the wasp33a values to the printed 1e-9 Ha (aarch64 vs x86, H100 vs Ada). Stability
analysis (§2.5 cases, C10 and C24, q0 and q1): same verdicts as CPU (all stable, no re-optimisation), energies within
2e-10 Ha.

**Noise floor:** `conv_tol` 1e-11 on circumcoronene q0 again did not converge in 200 cycles (Newton fallback, final
energy 2e-9 Ha from the production value); q1 converged in 25. The floor is not specific to the Ada card; keep
`conv_tol` 1e-9 with `conv_tol_grad` 1e-6.

### 5.3 Timings on one H100

Energy + forces + dipole, displaced geometry, production setting, after warm-up, one process:

| Molecule | nao | q0: s (cycles) | q1: s (cycles) | RTX 5000 Ada q0 / q1 (s, §2.3) | gain q0 / q1 |
|---|---|---|---|---|---|
| C10H8 | 166 | 11 (13) | 17 (21) | 12 / 19 | 1.1 / 1.1 |
| C24H12 coronene | 384 | 15 (15) | 27 (28) | 37 / 74 | 2.5 / 2.7 |
| C32H14 ovalene | 508 | 16 (15) | 29 (27) | 58 / 106 | 3.6 / 3.7 |
| C54H18 circumcoronene | 846 | 27 (18) | 41 (28) | 150 / 229 | 5.5 / 5.6 |
| C90H24 | 1398 | 59 (23) | 115 (46) | 452 / 793 | 7.7 / 6.9 |
| C144H30 | 2220 | 126 (25) | 246 (46) | 1123 / 1733 | 8.9 / 7.0 |

- Up to C32 the time hardly grows with size: overhead, not arithmetic, sets it (§5.4). The FP64 advantage
  (~30–60× peak, *sourced*) shows only from C54 up, as 5–9×.
- GPU memory ≤ 2.9 GB (C144 q1). Stability analysis on the H100 (pilot, §5.5): 33–105 s for C10–C48, 276 s for a C96
  cation; 1.4–4× the point's SCF + forces (neutrals, with the RKS→UKS test, cost more). CPU: 980–1590 s for C24 on 18 threads.
- Cation ⟨S²⟩ grows with size as on wasp33a (0.813 at C90, 1.040 at C144).

### 5.4 Several processes per GPU, MPS, and why

Ovalene, coronene and naphthalene (q0), 8 displaced frames per process, total points per hour:

| Molecule | 1 process | 2 | 4 | MPS: 1 | MPS: 2 | MPS: 4 |
|---|---|---|---|---|---|---|
| C10H8 | 364 | 374 | 360 | 370 | 352 | 381 |
| C24H12 | 237 | 325 | 329 | 242 | 312 | 313 |
| C32H14 | 215 | 288 | 309 | 203 | 271 | 299 |

- Unlike the Ada card, extra processes do help from C24 up: ~1.35× with 2 processes, up to 1.44× with 4. MPS (host
  daemon, 29 client connections logged) adds nothing on top. Capping OpenMP threads per process (4 × 16, 2 × 32)
  changed nothing. Naphthalene does not gain at all.
- Nsight Systems (one process, two points after warm-up): the GPU is busy only **13 %** of the wall time for
  naphthalene and **37 %** for ovalene. A point launches 80–220 thousand kernels with a median length of ~3 µs; the
  largest kernels are the J (`md_j_*`) and K (`rys_k_*`) builds. So a single SCF is launch- and host-bound, not
  FP64-bound, on the H100. Extra processes recover part of the idle time; that they stop at ~1.4×, with or without
  MPS, points to a shared host-side limit (driver launch path or per-process host work), not to the GPU.
- Consequence for a JAX/vmap port (§3): a batch that fused many molecules' kernels would attack exactly this idle
  time, so the case for batching is stronger on the H100 than on the Ada card. It is still not worth the 10–80
  person-weeks for the seed set (below); revisit if the full data set (50–80k points) runs on GH200s.
- Production: **2 processes per GPU, no MPS.**

### 5.5 Seed-set driver and pilot

`d1/dft_seed_run.py` is the GPU4PySCF counterpart of the Gaussian route (`g16_io.py`): same frames and frame names,
the same stability sample (the `stability_test` flags in `gjf/manifest.json`, 1,454 of 7,424; Stable=Opt for
cations, test only for neutrals), and `collect` writes the same extended XYZ as `g16_io.py parse` (`dft_energy` eV,
`dft_forces` eV/Å, `dft_dipole` e·Å about the centre of mass, `s2`, `stability`, `scf_cycles`, labels), so
`mlff_qa.py` runs on it unchanged. Resumable (one JSON per frame, written atomically), sharded over processes
(`d1/dft_seed_run.sh`, `GPUS`, `NPROC`).

Pilot (one process, GPU 0): C10H8_330, C24H14_204, C48H20_146 (32 frames per charge) and the transfer pair
C96H30_3317 (5 per charge), 202 frames, 33 with a stability test. 2.12 GPU-hours. Every SCF converged within 47
cycles, no Newton fallback, cation ⟨S²⟩ ≤ 0.786, all 33 stability tests stable; `mlff_qa.py` with the xTB
references quarantined nothing (24 boundary and 13 graph-changed frames labelled). Output:
`/var/tmp/jjlee-pah/d1/seed_dft/` (the pilot frames count toward the full run).

**Seed-set cost** (`d1/seed_cost.py`: series times interpolated in nao, scaled by the pilot's measured/predicted
ratio 1.06; stability tests at the pilot's 2.8× a single point): 57.6 GPU-h of single points + 29.0 GPU-h of stability
tests = **≈ 87 GPU-hours on one H100**, i.e. **≈ 33 h on the node's two GPUs** with 2 processes each (~1.3×).
*Estimated from measured times.* Against §2.3: ~280 RTX 5000 Ada hours without stability tests.
