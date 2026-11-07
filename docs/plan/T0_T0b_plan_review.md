# Readiness review: T0/T0b Execution Plan on GH200

Reviewed 2026-11-06 against the 14 references in `corpora/references/` and the current state of the software packages (PyPI and GitHub, checked the same day).

**Status (2026-11-06):** every item in sections A–D has been applied to `T0_T0b_execution_plan_GH200.md`. Still to confirm: the exact container image tags (the NGC PyTorch tag is left as a placeholder).

**Verdict before the fixes: not ready as written.** The hardware and benchmark design are sound. However, two T0b pass criteria cannot be met as stated. Several configuration snippets would fail without an error message, or not run at all. Most fixes are edits to the text, not redesigns.

Section numbers below refer to `T0_T0b_execution_plan_GH200.md`.

---

## A. Blockers: a criterion fails, or code silently does the wrong thing

### A1. The Orion Bar ionisation-order check contradicts the data (§1, §5.10, §5.11)
The plan expects ionisation fraction to "decrease monotonically … into the molecular zone (DF3 lowest)". Chown et al. 2024 (§4.3) report the opposite:

- 6.2/11.2 is **12% *higher* in DF3** than at the PDR surface.
- The variation is called "limited" and "somewhat surprising", given a ×40 change in ionising flux.
- The AIB emission in the H II-region template "originates from the background PDR".
- DF1 resembles the atomic-PDR template because of foreground extinction. Peeters et al. 2024 give the internal extinction per template.

Any formula that increases with r62 will therefore put DF3 *most* ionised. Even then, a 12% effect is close to the slice's systematics.

**Fix.** Replace the pass/fail order test with consistency checks:

- Predicted fractions lie in [0, 1].
- The five templates agree with each other within their propagated uncertainties.
- For the atomic PDR, the prediction is compared with the PAHdb-fit charge breakdown of the same spectrum in Ricca et al. 2025 (§6, Fig. 8).
- Report the template order as an observation, not as a gate.

### A2. The formula criterion contradicts the rediscovery goal (§5.9, §5.11)
- **Error cap.** The criterion asks for test error ≤ 0.5 × the one-ratio fit. But a rediscovered "6.2/11.2-type" formula that is a monotone function of r62 alone cannot beat the best smooth 1-D fit in r62. That fit *is* the one-ratio baseline. The criterion can only pass with a formula that uses two or more ratios.
- **Complexity cap.** The plan's own example, `a*r62/(b + r62)`, has PySR complexity 7, which exceeds the cap of 6.

**Fix.** Pass if either of these holds:

- **Single-ratio formula:** monotone in a 6–9 µm ratio, with error ≤ 1.1 × the one-ratio fit.
- **Multi-ratio formula:** error ≤ 2 × the forest ceiling.

Raise the complexity cap to 7, or replace the example.

### A3. PySR configuration (§3.3 container C, §5.8): checked against PySR 2.7.2 source
- **Weights are silently dropped.** `weights=1/sigma_y**2` in the `PySRRegressor(...)` constructor is ignored with only a `FutureWarning`, so the fit runs unweighted. Pass it to `model.fit(X, y, weights=w, variable_names=[...])`.
- **`pysr.install()` has been removed.** It is now a no-op that warns. Julia is installed through `juliacall` on first import, so the `juliaup` line is unnecessary. To bake Julia into the image, run `python -c "import pysr"` during the build.
- **`procs=16` has no effect** together with `parallelism="serial"`. For the seed runs, use `parallelism="multiprocessing", procs=16`.

### A4. The synthetic dust continuum is unphysical (§5.5)
The plan uses a 60 K modified blackbody (ν² emissivity) normalised to 30% of the 11.2 µm peak. Relative to its value at 11.2 µm, that curve is:

- 3.9 × 10⁻⁴ at 7.7 µm (effectively zero below 8 µm)
- 53× at 15 µm
- **680× at 20 µm**, about 200× the 11.2 µm band peak

That swamps 12–20 µm and will dominate PAHFIT fit failures.

**Fix.** Use warmer components, such as PAHFIT's fixed set (35–300 K; Smith et al. 2007, §4.1.2), or a continuum shape taken from the Orion templates. Apply the random scaling to those components.

### A5. The `r110` feature does not exist in PAHFIT (§5.6, §5.7, §5.8)
PAHFIT 2.0 has one science pack, `classic.yaml`, which is the Spitzer-era Smith 2007 model. It has no 11.0 µm feature: in that range there are only 10.68, 11.23 and 11.33 µm.

- So r110 will be empty. Either drop it, or add an 11.0 µm component to a custom science pack.
- "JWST-era default science pack" should read: `classic.yaml` plus the `jwst.yaml` instrument pack.

### A6. Container recipes will not build as written (§3.3)
- **Version requirements.**
  - JAX 0.12 needs Python ≥ 3.12, and mace-jax caps JAX below 0.12 for its cuEquivariance backend.
  - mace-torch 0.3.17 needs Python ≥ 3.11.
  - The `nvcr.io/nvidia/pytorch:24.09-py3` image ships Python 3.10, so `pip install mace-torch` will fail or fall back to an old release.
- **Interpreter mismatch in container A.** It installs `python3.11`, but bare `pip` points to the system Python 3.10. Ubuntu 22.04 has no `python` command, so the verification line fails.

**Fix.**
- Use Ubuntu 24.04 or Python 3.12 base images and a newer NGC PyTorch tag.
- Install everything into one venv per container, with `python -m pip`.
- Install mace-jax with its `[jax-cuda12]` or `[jax-cuda13]` extra rather than installing JAX separately.

---

## B. Should fix before starting

### B1. cuEquivariance is no longer the ARM risk, but it is now a fairness variable (§3.3, §4.5, §7)
- Version 0.13.0 has aarch64 wheels for both `cuequivariance-ops-torch-cu12` and `cuequivariance-ops-jax-cu12`.
- mace-jax also uses the cuEquivariance backend.
- Run a 2 × 2 grid: each stack with and without cueq. State in the memo which combination the decision rests on.

### B2. The PyTorch baseline answers the wrong objection (§4.3, §4.4)
- **Sequential runs add no information.** Running batch 100 sequentially is just 100 × batch 1. MACE-OFF reports about 9.6 × 10⁶ steps/day for a ~33-atom molecule in vacuum on an A100 (Kovács et al. 2025, §3.4.2), and ASE adds overhead. That puts sequential batch 100 at roughly 10+ GPU-hours per repeat.
- **Better baseline:**
  - Measure batch 1 and extrapolate.
  - Add a concurrent baseline: N ASE processes sharing the GPU through CUDA MPS, using the 72 Grace cores. That is the realistic way to fill an H100 with small molecules in PyTorch.

### B3. Inconsistent equivalence tolerance
The smoke test (§3.4) uses about 1e-4 eV/Å, while §4.2 uses 1e-3 eV/Å. Pick one; 1e-3 eV/Å in float32 is realistic.

### B4. The relation to Mai et al. 2025 is overstated (§2, §8 item 6)
- **What Mai et al. actually used:**
  - A **NeuralIL** force field (written in JAX), not MACE, trained on B3LYP/4-31G data.
  - An ASE driver.
  - Production runs of 200 ps Nosé–Hoover NVT followed by 200 ps NVE at 0.5 fs, sampled at 1 fs.
- **What T0 benchmarks:** a stack in the style of Mai's workflow. Say so in the reading list.
- **Optional:** time NVE as well, since that is the production ensemble.
- **Project-level caveat, not a T0 blocker:** MACE-OFF is trained for neutral, closed-shell systems only (Kovács et al. 2025, §4, on SPICE ωB97M-D3(BJ)/def2-TZVPPD data). It cannot produce cation trajectories for the later stages.

### B5. Species selection (§4.1, §5.1)
- **Coronene has only duo CH, not solo.** Compact PAHs below ~C50 are duo-dominated; solo CH dominates only for the largest compact species. Redefine the compact class in those terms.
- **Cations are scarce in the smallest bin.** PAHdb v4.00 holds 786 neutral but only 30 cation pure PAHs with 21–30 C (Ricca et al. 2025, Table 1). Select neutral/cation *pairs* first.
- **Solo/duo/trio counts are not stored directly.** Derive them from the boundary-edge code; the AmesPAHdbPythonSuite computes it (Ricca et al. 2025, §2.4 and §3.2).
- **§4.1 range mismatch.** Pyrene (C16) is below the stated 20–30 C bin.

### B6. Wang 2026 is mischaracterised (§8 item 9)
- **Correct reference:** Wang 2026, *Full-spectrum infrared fingerprinting: A transformative AI paradigm for interstellar polycyclic aromatic hydrocarbons*, A&A 710, L17.
- **What it actually does:**
  - It trains a random-forest **classifier** on full spectra binned at 20 cm⁻¹, using about 23,000 spectra.
  - It sorts them into 12 size × charge classes, with F1 ≈ 0.963.
  - It does **not** regress ionisation fraction from band ratios.
- **Lesson for T0b (its Fig. 1):** ratio diagnostics can look good purely because of sample selection. I₁₁.₂/I₃.₃ against N_C gives R² = 0.82 on 81 selected molecules, but only 0.23 on 15,022.
  - A hand-picked set of 25 species is exactly that situation, so the slice report should say the rediscovery result is optimistic.

### B7. Data sources (§3.5): make them concrete
- **MIRI templates:** CDS J/A+A/685/A75 (Chown et al. 2024).
- **NIRSpec templates:** CDS J/A+A/685/A74 (Peeters et al. 2024).
- **Apertures and cubes:** MAST HLSP, doi:10.17909/wqwy-p406.
- **MRS resolving power:** use the polynomials in PAHFIT's `jwst.yaml`, so that simulation and fitting share one instrument model. Argyriou et al. 2023, Eq. 1, is the published reference curve.

### B8. Schedule (§6)
The T0 memo is due in week 5, but the batched-PyTorch baseline is scheduled for week 6. Either move the baseline into week 5, or make the week-5 memo a draft.

---

## C. Minor
- **Cascade photon energy (§5.3).** 6 eV matches Wang 2026. Ricca et al. 2025 use 8 eV for the Orion atomic PDR. State the choice as a known systematic for §5.10.
- **Dipole proxy (§4.3).** "Centre-of-charge displacement" needs per-atom charges, and MACE does not provide them. Specify fixed charges, for example ±0.1 e on C–H pairs, or just save positions.
- **PAHFIT speed (§5.6).** "About a minute per spectrum" is unverified. Time 20 fits on day 4 before committing to the 64-core estimate.
- **Unused package (§3.3 container C).** `pysindy` is installed but never used.
- **GH200 memory (§3.2).** GH200 nodes come with 96 GB HBM3 or 144 GB HBM3e; read the actual value from `nvidia-smi`.

## D. Reading-list corrections (§3.1, §8)
| # | Change |
|---|---|
| 2 | Full title: *PDRs4All IV. An embarrassment of riches: Aromatic infrared bands in the Orion Bar*, A&A 685, A75 (2024) |
| 2b | **Add** Peeters et al. 2024, *PDRs4All III*, A&A 685, A74: template apertures and extinction |
| 3 | **Add** Ricca et al. 2025, ApJS 282, 7 (PAHdb v4.00, doi:10.3847/1538-4365/ae1c38). The plan uses v4.00 but cites only the v3.00 paper |
| 5b | **Add** Kovács et al. 2025, *MACE-OFF*, JACS 147, 17598 (doi:10.1021/jacs.4c07099): the model T0 actually runs |
| 5c | Optionally cite MACE-MP-0 as Batatia et al. 2025, J. Chem. Phys. (published version) |
| 6 | Mai et al. 2025: reword as described in B4 |
| 9 | Wang 2026: correct the title and description as described in B6 |
| new | **Add** Argyriou et al. 2023, A&A, *JWST MIRI flight performance: The Medium-Resolution Spectrometer*: the MRS resolving power used in §5.5 |
