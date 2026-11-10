# Committee review: T0/T0b Execution Plan on GH200

Reviewed 2026-11-06 against `T0_T0b_execution_plan_GH200.md`, after the fixes from `T0_T0b_plan_review.md` had been applied.

**Committee.** Voting members: Fable 5.1 and GPT-6-Astra. A third reviewer, Opus 5.5, wrote a round-1 review that both members read; it did not vote.

**Method.** Three rounds:
1. Independent reviews.
2. Positions on eight disputed points.
3. Final positions on three points that were still open.

**Status (2026-11-06):** every Blocker, Should-fix and Minor item below has been applied to `T0_T0b_execution_plan_GH200.md`. The only item still open is the NGC tag: the plan pins `26.10-py3`, which should be confirmed against the support matrix before the first build.

**Verification round (2026-11-06):** both members checked the revised plan against their reviews. Fable reported 12 issues and Astra 14, overlapping on four:
- the converter defaults to the *medium* MACE-OFF model unless `--model-name small` is given;
- the endpoint constants need band strengths, not just ratios;
- the rediscovery gate was over-strict;
- Ricca's fit covers 6–15 µm.

All the issues have been applied to the plan. Two of them correct this consensus document itself:
- **Endpoint algebra (Should-fix 2 and 3 above):** the exact-inverse check now uses endpoint band strengths at fixed size distributions, and is judged on recovering f.
- **Ricca comparison (Should-fix 1):** a numerical comparison needs an auxiliary predictor of the emission label; the number-fraction formula cannot be compared directly.

The two members disagreed on the facts of each library's default centre-of-mass handling. The plan states only ASE's default and requires one explicit policy across all modes.

**Evidence.** Literature claims are grounded in the papertrail corpus (`corpora/references`). Software claims were checked against PyPI, GitHub and vendor documentation on 2026-11-06.

## Verdict

**Not ready to hand over. The fixes are text edits, not a redesign.**

The overall design is sound:
- the 2×2 cuEquivariance grid;
- neutral/cation pairs;
- treating the order of the Orion templates as an observation, not a gate.

Four things must change before the student starts:
1. the PAHFIT install;
2. the PyTorch baseline at batch 100;
3. the uncertainty-dependent deliverables;
4. precision parity in T0.

---

## Blockers

### 1. `pip install pahfit` installs the wrong PAHFIT (§3.3 container C, §3.4, §5.6)

**Problem.** PyPI has one release, 2.0, from February 2022. It ships `scipack_classic.yml` and the old API: there is no `classic.yaml`, no `instrument/jwst.yaml` and no `Model.from_yaml`. The pack-based API exists only from tag v2.5 (2023) and on the trunk. The import name is unchanged, so nothing fails until §5.6.

**Sources:** https://pypi.org/pypi/pahfit/json; https://github.com/PAHFIT/pahfit/tree/v2.0/pahfit/packs; https://pahfit.readthedocs.io/en/latest/install.html.

**Fix.**
- Install PAHFIT from git at a pinned commit, and record the commit hash as "the PAHFIT version" in the §5.11 provenance.
- Smoke test 4 must call `Model.from_yaml('classic.yaml')` on a spectrum whose instrument is a `jwst.miri.mrs.*` segment.
- Also install and pin AmesPAHdbPythonSuite in container C.
- Make sure the conversion step has `mace-torch` available (container A, or a dedicated conversion environment).

### 2. The PyTorch baseline at batch 100 cannot run as written (§4.3–4.5, §6)

**Problem.** "*b* independent ASE processes through MPS" means 100 simultaneous clients, which is over the MPS limit. A client that cannot connect silently runs without MPS. The MPS limit is:
- **60 client contexts per device** on the Release 620 driver with the default `CUDA_DEVICE_MAX_CONNECTIONS=2`;
- 48 on older drivers and man pages.

**Sources:** https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html#client-server-connection-limits; https://docs.nvidia.com/deploy/mps/latest/architecture.html.

**Fix.**
- **Separate workload from concurrency.** The batch of 100 is the total number of trajectories. Run P ≤ 48 workers, with P also below the core count, each cycling through ⌈100/P⌉ molecules.
- **Record what actually ran.** Note the installed driver's limit and P, and confirm in the MPS server log that all P clients attached.
- **Add a protected batched-PyTorch mode,** run in weeks 4–5 with the same weights, precision and physics settings. A GH200 pilot in week 4 decides between two options:
  - **TorchSim** via MACE 0.3.17's first-party `MaceTorchSimModel`. Pass `dtype=torch.float32`; its default is float64.
  - **Concatenated ASE `Atoms`:** all molecules in one system, spaced ≥ 30 Å apart, with an end-of-run assertion on the minimum distance between molecules.
- **Pilot pass conditions:**
  - the option installs on aarch64;
  - its batched forces match isolated ASE+MACE to within 1e-3 eV/Å;
  - no edges between molecules appear in the neighbour graph;
  - a 100-molecule, 1 ps run completes.
- **Report which stack it was.** The memo must say whether the conclusion concerns ASE+MACE or TorchSim+MACE.
- **Protect it in §6.** The selected batched mode and the MPS pool are never cut. Cut the cuEquivariance-off runs first, because §4.5 compares JAX against the fastest PyTorch mode measured.

### 3. PAHFIT 2.x returns no fit uncertainties (§5.6, §5.8, §5.10, §5.11)

**Problem.** The docs say "Fit uncertainties will be supported like this in the future." Issue #149 has been open since 2021, and PR #313 ("covariance uncertainty output helpers") was closed unmerged in June 2026. Yet the plan relies on "plus their uncertainties", `sigma_y`, propagated template uncertainties and an uncertainty-based Orion gate.

**Fix.**
- **Estimate uncertainties by Monte Carlo:** perturb the spectral noise and refit fully, as in Ricca2025_ApJS_PAHdb_v4, Appendix, "Monte Carlo Fitting Approach" (1024 samples there).
  - Compute complex sums, ratios and formula predictions inside each realisation.
  - Apply this to all Orion templates and to a representative synthetic validation subset.
  - Start at 30–50 refits as a pilot and increase until the intervals are stable.
- **Systematics:** use the per-band recovery bias from the injection test (Should-fix 2) as the systematic.
- **PySR:** run it **unweighted**. The label (the ionisation fraction from the Latin hypercube) is exact, and the noise is in the features, so `1/sigma_y**2` is ill-defined. Delete the `w` line in §5.8 and the "unweighted loss" advice in §7.
- **Severity:** this item is a Blocker as long as uncertainty-dependent deliverables remain in the plan. Once the Orion criteria are rewritten as in Should-fix 1, it drops to Should-fix.

### 4. T0 does not enforce precision parity (§4.2–4.4)

**Problem.** "Use float32 in both" appears only in the equivalence step, while the library defaults go the other way:
- `mace_off()` defaults to `default_dtype="float64"`; `mace_mp()` defaults to float32. Source: `mace/calculators/foundations_models.py`.
- `MaceTorchSimModel` defaults to float64.
- JAX float32 matrix multiplies use TF32 "where available" on the H100. Source: https://docs.jax.dev/en/latest/201/precision.html.
- NGC PyTorch images are reported to set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`. This is attested by MONAI's documentation, not NVIDIA's own notes.

Any of these can silently move the 5× decision, and TF32 can break the 1e-3 eV/Å force gate for a reason §7 does not list.

**The converter has no dtype option.** `mace-jax-from-torch` takes only `--torch-model`, `--foundation {mp, off, anicc, omol}`, `--model-name` and `--output`. (`--torch-param-dtype` belongs to `mace-jax-train`.) MACE-OFF conversion is supported through `--foundation off`.

**Fix: add a "precision parity" row to §4.4.**
- Pass `default_dtype="float32"` (or `dtype=torch.float32` for TorchSim) explicitly.
- Cast the converted parameters to float32 when the JAX driver loads them, and assert that `jax_enable_x64` is off.
- Set `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- In container B:
  - unset `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`;
  - call `torch.set_float32_matmul_precision("highest")`;
  - set `torch.backends.cudnn.allow_tf32 = False`;
  - make sure no autocast is active.
- Print the effective settings at start-up and store them in the benchmark CSV.
- Report the decision at matched precision. Any TF32 runs go in a separate column.

**Severity.** Fable rates this a Blocker; Astra rates it a Should-fix that "must be completed before any timing". Both require it before any timing counts.

---

## Should-fix

### 1. Replace Orion criterion (2) (§1, §5.10, §5.11)

**Problem.** "The five templates agree within propagated uncertainties" assumes the same ionisation fraction in all five zones, and the evidence does not support that:
- Chown2024_AA_PDRs4All_IV §4.3 calls the 12% variation in 6.2/11.2 "somewhat surprising", raises a possible cation contribution at 11.2 µm, and leaves the interpretation to further modelling.
- Statistical errors on these spectra are ≤3×10⁻⁴, and the systematics are "currently unquantified" (Ricca2025_ApJS_PAHdb_v4, Table 3 notes).
- So the gate either fails on any real difference, or it gets widened by hand until it passes.

**Fix: the Orion application reports the following, and none of it is a pass/fail gate.**
- Coverage of each template by the joint training-feature distribution (not only per-feature percentiles); anything outside is flagged as extrapolation.
- The unclipped predictions, and whether they lie in [0, 1].
- The spread between templates, next to the held-out prediction error and the Monte Carlo measurement uncertainty: "Δf between templates is X; the slice resolves Y."
- For the atomic PDR, a comparison of the *emission-weighted* cation fraction with Ricca2025 Table 3: 64.86% cation at v4.00. Fig. 8 is in the Appendix, not §6. List the known differences:
  - 8 eV photons versus 6 eV;
  - Gaussian 20/15/10 cm⁻¹ profiles with no redshift, versus Lorentzian 15 cm⁻¹;
  - the 6–15 µm range;
  - the molecular pool (N_C 21–384, including nitrogenated PAHs);
  - gas lines in the templates, such as [Ne II] 12.81 µm on the 12.7 µm complex.
- An out-of-domain or unphysical result is a transfer failure to investigate. It does not by itself mean the plumbing is broken.

### 2. Present the rediscovery as a controlled closure test (§1, §5.5–5.9, §5.11)

**Problem.**
- **Monotone by construction.** At fixed size weights, the population spectrum is linear in f, so any band ratio is monotone in f. Passing therefore rules out only gross errors.
- **Continuum in the fit basis.** The synthetic continuum is built from PAHFIT's own fixed blackbody set (Smith2007_ApJ_PAHFIT §4.1.2), so the fit model represents it exactly. A continuum bug cannot show up, even though §5.10 names the continuum as "the usual culprit".
- **Test set used for selection.** The plan chooses the front member by test error.

**Fix: rediscovery passes only if all of the following are run and reported.**
1. **Injection–recovery.** On noise-free, continuum-free populations, compare PAHFIT band strengths with direct integration of the input spectrum, band by band. Record the bias; this also measures the Lorentzian-versus-Drude profile mismatch, which the plan currently calls "harmless".
2. **Analytic endpoints.** Fit pure-neutral and pure-cation versions of each population, plus a few known intermediate mixtures. Their ratios fix the constants of the exact inverse, which the single-ratio PySR member must reproduce.
3. **Out-of-basis continuum.** Repeat on a 200-spectrum subset with a power law or the atomic-PDR template's continuum shape.
4. **Shuffled-label negative control.** One extra PySR run, which must fail.
5. **Validation-only selection.** Choose the formula, seed and smoother settings on the validation 15%, then evaluate the locked choice once on the test 15%.

### 3. PySR criterion and complexity (§5.7, §5.9, §5.11)

**The exact form fits under the cap.** The exact single-ratio inverse of a linear mixture, k·(r−a)/(b−r), rewrites as c1 + c2/(c3 − r). That has complexity 7 under PySR's node count (Cranmer2023_arXiv_PySR §2.1), exactly at the cap.

**Fix.**
- Keep `maxsize=12` and the ≤ 7 simplicity target, and report the front up to complexity 9.
- State the expected form c1 + c2/(c3 − r62), with constants from the endpoint check.
- **Pass conditions:**
  - *Single ratio:* a monotone member whose test error is ≤ 1.1× the one-ratio baseline, which also beats the constant predictor and meets a pre-specified absolute accuracy target. The one-ratio baseline must be pre-specified and tuned on validation: isotonic regression, or a monotone spline with fixed knots.
  - *Multi-ratio:* error ≤ 2× the forest *reference* and below the one-ratio fit.
- **Wording:**
  - Drop "cannot beat": a fitted smoother is a comparator, not an optimum.
  - Call the forest a reference, not a ceiling.
  - A failure is a diagnostic outcome, not automatic proof that the plumbing is broken.

### 4. The "12.7/11.2 means a leak" diagnosis is wrong (§5.9, §7)

**Problem.** "The 12.7 AIB complex is due to either duo H's in neutral PAHs or trio H's in cations" (Chown2024_AA_PDRs4All_IV §3.6). Tielens2008_ARAA §3.5 likewise treats the band as tracing structure. So r127 legitimately carries charge information, and §5.9 contradicts itself: it accepts "r62 corrected by r127" as a pass, then calls reliance on r127 a leak. The remedy in §7 is also vacuous, because the Latin hypercube makes f and the mean N_C independent by construction.

**Fix.**
- Delete the leak claim.
- If r127 matters, refit without it and report how much the error grows.
- The real leak risk is in the species list: check that the cation/neutral pairs span the same N_C range in each compactness class.

### 5. Weak 11.2 µm denominator for duo-only compact species (§5.1, §5.7)

**Problem.** The 11.2 µm band comes from solo C–H on neutral PAHs (Bauschlicher2018_ApJS_PAHdb_v3 §4: solo, duo, trio and quartet H at 11.2, 12.0, 12.7 and 13.5 µm). Cations contribute only through the 11.0 µm satellite (Ricca2025 Table 3 notes). Compact PAHs below about 50 C are duo-only, so small-size populations may have a weak or misplaced 11.2 µm band, and ratios with a near-zero denominator get heavy tails.

**What not to do.** Adding a species-level selection rule (for example "at least two solo H") was rejected, because it repeats the selection effect that Wang2026_AA_L17 Fig. 1 warns about.

**Fix.**
- Record the solo, duo and trio counts per species in `species.csv` (derived with the suite's `Geometry.bec()`), and report the fraction of species without solo H in each size bin.
- In week 3, plot the population-level 11.2 µm strength relative to total PAH emission across the size × charge grid, including the endpoints. Do not rely on S/N alone: the noise is set at the 11.2 µm peak, which hides intrinsic weakness.
- Before PAHFIT runs, predefine the domain where 11.2-normalised ratios are valid.
- If weak denominators are widespread, make the total-PAH-normalised features primary, and report the classical ratios only inside the declared domain.
- Never drop populations silently.

### 6. Define the ionisation-fraction label (§5.4)

**Problem.** "Weights from the size distribution and ionisation fraction" does not say whether the label counts molecules, carbon atoms or emitted power. A log-normal centred near 100 C with width 0.2 dex, laid over a discrete 20–110 C ladder, puts about half its weight beyond the largest species.

**Fix.**
- Weights are the log-normal density × bin width per species, renormalised after truncation.
- State whether UV absorption scales with N_C.
- Store the number fraction and the emission-weighted fraction as two labels.
- Store the realised size moments.

### 7. T0 trajectory check cannot catch mass or time-unit errors (§4.3, §4.4)

**Problem.** Mean temperature and its variance do not depend on mass, by equipartition. jax-md's `nvt_langevin` defaults to `mass=1.0` and has no fixed units: in eV, Å and amu one time unit is about 10.18 fs, so dt = 0.5 fs is 0.0491 and γ = 0.01 fs⁻¹ is about 0.102. mace-jax exposes padded jraph graph batches, not a ready-made energy function for use under `vmap`.

**Fix.**
- Pass the masses explicitly, convert dt and γ explicitly, and exclude padded atoms from the temperature.
- Build `md_jax.py` around one padded batched graph, with overflow detection.
- Add these checks:
  - padded-versus-unpadded force agreement;
  - a 1 ps NVE energy-drift test per stack;
  - a comparison of the vibrational density of states (C–H stretch near 3050 cm⁻¹ in both stacks).
- Wait for warm-up to finish (block on its results) before starting the timer.
- Budget week 3 for the JAX energy function.

### 8. Container B: CUDA generation and cuEquivariance pins (§3.3)

**Problem.**
- NGC PyTorch has shipped CUDA 13 since 25.08; every 26.xx tag is Python 3.12, CUDA 13 and multi-arch. So `-cu12` ops packages mismatch any tag the plan would pick.
- mace-torch 0.3.17 caps the cuEquivariance ops packages below 0.11 and `cuequivariance-torch` below 0.12, and mace-jax pins 0.10.x. Unpinned installs resolve to 0.13, outside both ranges.

**Sources:** https://github.com/ACEsuit/mace/blob/v0.3.17/setup.cfg; https://github.com/ACEsuit/mace-jax/blob/main/setup.cfg; https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html.

**Fix.**
- Pin a concrete tag, for example `26.10-py3`.
- Install `mace-torch[cueq,cueq-cuda-13]` constrained to cuEquivariance 0.10.0, which has aarch64 wheels.
- Freeze NGC's torch with a constraints file, run `pip check`, and assert that `torch.__version__` is unchanged after the install.
- In both stacks, print which backend MACE reports and record the cuEquivariance version per stack.
- State that container B deliberately uses the NGC interpreter and no venv.

### 9. Container C: the Julia depot is not writable at run time (§3.3)

**Problem.** The text says to set `JULIA_DEPOT_PATH`, but the recipe never does. The SIF image is read-only at run time.

**Fix.**
- Before `import pysr` in `%post`, set `JULIA_DEPOT_PATH=/opt/julia_depot` (and a fixed juliapkg project path).
- At run time, put a writable depot first: `JULIA_DEPOT_PATH=$SCRATCH/julia:/opt/julia_depot`.
- Consider offline mode. (The `PYTHON_JULIAPKG_OFFLINE` variable name is unverified.)
- Test as a non-root user with `--containall`. PySR's repository ships an `Apptainer.def` to start from.

---

## Minor

- **§4.3:** "Section 5 buckets species by atom count" is wrong; bucketing appears only in §7. Decide on bucketing in §4.4.
- **§1 vs §5.11:** §1 says "6.2/11.2 or 7.7/11.2", while §5.11 says "a 6–9 µm ratio", which also admits 8.6. Align them.
- **§6:** the embedded schedule chart is a broken placeholder.
- **§5 vs §6:** the day labels conflict with the week schedule. Say that day numbers are effort, not calendar days.
- **§4.4:** `wall/(10 ps × b)` is the cost per molecule-picosecond, whose reciprocal is aggregate throughput. Report aggregate and per-trajectory speed separately.
- **§6:** move a one-molecule conversion smoke test into week 1, since smoke test 3 needs it.
- **§5.3:**
  - State that no redshift is applied, following Ricca 2025.
  - Consider Gaussian 15 cm⁻¹ profiles for comparability.
  - Record that the 6 eV photon energy lies between Wang 2026 Appendix B (about 3 eV) and Ricca 2025 (8 eV).
- **§5.5:** resample onto the Orion template wavelength grid (CDS J/A+A/685/A75). The coefficients in `jwst.yaml` are pre-flight estimates (Labiano 2021), not Argyriou2023_AA_MIRI_MRS Eq. 1. Sharing them gives consistency, not independent validation.
- **§5.6:** PAHFIT reports individual features. Sum the 7.7, 11.3 and 12.6 µm complexes by the group keys in `classic.yaml`, and check those keys once.
- **§5.11:** the four-hour `make slice` budget must cover 2,000 PAHFIT fits and four PySR runs. Allow cached stages.

## Corrections to the previous review (`T0_T0b_plan_review.md`)

| Item | Correction |
|---|---|
| A1 | Rejecting the order test was right, but replacing it with mutual agreement within uncertainties overcorrected (Should-fix 1). |
| A2 | The one-ratio fit is not an unbeatable bound, and the example `a*r62/(b+r62)` assumes neutrals emit no 6.2 µm band. The exact form is c1 + c2/(c3−r) (Should-fix 3). |
| A3 | Moving `weights` into `fit()` fixed the API call, but the weight itself is ill-defined, and PAHFIT gives no sigma (Blocker 3). |
| A4 | PAHFIT's own temperature set fixed the swamping problem but put the continuum inside the fit basis (Should-fix 2). |
| A5 | It describes the PAHFIT trunk, not the 2.0 release on PyPI (Blocker 1). |
| B1 | cuEquivariance 0.13 installs, but it is outside both MACE stacks' pins; use 0.10.x (Should-fix 8). |
| B2 | The MPS pool ignored the client limit, and the plan left batched PyTorch as the first item to cut (Blocker 2). |
| B5 | The duo-only point was applied to the wording only; its effect on the 11.2 µm denominator was not followed through (Should-fix 5). |
| — | The review missed the precision defaults, the jax-md mass and unit defaults, and NGC's switch to CUDA 13. |

## Where the committee diverged and how it resolved

| Point | Round 1 | Resolution |
|---|---|---|
| Orion agreement gate | Fable silent; Astra and Opus opposed | Dropped (all) |
| MPS limit | 48 (Fable, Opus) vs 60 (Astra) | 60 on current drivers, 48 on older ones; record the installed limit |
| Missing uncertainties | Blocker (Fable) vs Should-fix (Astra) | The two swapped positions in round 2; settled on a conditional Blocker |
| cuEquivariance | 0.10 (Fable) vs 0.13 OK (Opus) | 0.10.x (both members) |
| Complexity cap | Opus: the exact form needs complexity 9 | Rewritten as complexity 7; the cap stays |
| 11.2 µm denominator | Solo-H rule (Fable) vs population check (Opus); Astra silent | Population-level check plus recorded H-adjacency counts; no species rule |
| `--torch-param-dtype` | Fable cited it for the converter | It doesn't exist there (it belongs to `mace-jax-train`); cast at load |
| Batched PyTorch primary | ASE concatenation (Fable) vs TorchSim (Opus, Astra) | Each member conceded to the other's original choice in round 3; the week-4 pilot decides, with explicit criteria |
| Precision severity | Fable Blocker, Astra Should-fix | Unresolved label only; both make it mandatory before timing |

## Unverified items

- Whether TorchSim and `nvalchemi-toolkit-ops` run on aarch64 (to be checked in the week-4 pilot).
- The `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` setting in NGC images (attested by a third party only).
- The `PYTHON_JULIAPKG_OFFLINE` variable name.
- That duo-only species actually give weak 11.2 µm strength in PAHFIT fits. This is reasoned from the band assignments, not fitted.
- Whether jax-md 0.2.29 and mace-jax are compatible on the same e3nn-jax release.

**Status 2026-11-07** (measured on the project GH200; see `GH200_feasibility_check_2026-11-07.md`):
- TorchSim and `nvalchemi-toolkit-ops`: *install and batched forces verified* on aarch64 + H100, with PyPI torch 2.15.0+cu132: 6.5e-7 eV/Å against per-molecule ASE. Re-checked inside the NGC 26.10 image the same day: 9.7e-7 eV/Å, and a 100-molecule batched Langevin run executes. Later that day the 1 ps, 100-molecule pilot passed all four §4.3 criteria in the NGC image, with units, temperature (598.7 K) and the C–H stretch (3169 cm⁻¹) checked; the force call turned out host-bound (feasibility record §H). The PAHFIT synthetic closure fit also passed, but PAHFIT's default `maxiter=1000` leaves fits unconverged without error (§G).
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` in NGC images: *verified* from the registry config of the `26.10-py3` arm64 image.
- `PYTHON_JULIAPKG_OFFLINE`: *verified*. juliapkg reads `PYTHON_JULIAPKG_` + option name. An offline `import pysr` worked and wrote `lock.pid` into the project directory, confirming the writable-project requirement.
- jax-md 0.2.29 and mace-jax on the same e3nn-jax: *verified at smoke level*. Both ran in one environment on e3nn-jax 0.21.0 (jax 0.11.2).
- Duo-only 11.2 µm strength: still open (science, not environment).

## Addendum 2026-11-08: the T0 energy gate

The §4.2 equivalence check was run on the project GH200 on 2026-11-07 (20 PAHdb v4.00 species, 120 frames; `GH200_feasibility_check_2026-11-07.md` §I). Forces passed in every cuEquivariance combination (max 2.3e-5 eV/Å). The float32 energy gate of 1e-4 eV/atom failed (cross-stack 5.8–6.5e-4 eV/atom), and each stack was further from its own float64 result (9e-4 eV/atom) than from the other stack. Both stacks in float64 differed by 2.1e-5 eV/atom (3.0e-3 eV total). The run's author proposed moving the energy comparison to float64 (Claude Doc "T0 equivalence gate: proposed energy-criterion change"). Fable and Astra reviewed it over three rounds.

**Findings.**
- The float32 failure comes from the total energy: MACE totals include the atomic reference energies (−2e4 to −1e5 eV), whose float32 spacing is close to the whole per-atom budget. Forces derive from the interaction energy only, so they never see this floor.
- `ScaleShiftMACE` in both MACE and mace-jax sums `interaction_energy` separately, before adding E0, so it can be gated at float32. MACE's ASE calculator does not return it; take it from the model directly.
- The float64 residual is explained by the converter: `mace-jax-from-torch` builds its template in JAX's default dtype (float32 unless x64 is on) and casts every floating torch tensor to it, so the float64 checkpoint was rounded at conversion. The earlier instruction to cast to float32 at load remains right for benchmarking, but it omitted preserving float64 precision during conversion; later promotion cannot recover precision already lost.
- Round 2's claim that MACE's cuEquivariance modules are built under torch's global float32 default was wrong: `convert_e3nn_cueq.run` sets the default dtype from the source model. The mace-jax cuEquivariance math dtype remains unverified.

**Adopted (plan §4.2, with §3.4 smoke test 3, §4.1, §4.4 Precision and Correctness rows, §6 week 3).** Three gates: float32 forces < 1e-3 eV/Å and float32 interaction energies < 1e-4 eV/atom for every cuEquivariance combination, with `ScaleShiftMACE` asserted; float64 totals < 1e-6 eV/atom from an x64 conversion, with cuEquivariance off and on, verified effective math dtype, and "unresolved" if float64 cannot be shown. Float32 totals are reported, not gated. Failures are diagnosed by decomposition (Σ_Z n_Z ΔE0_Z, constancy across geometries of one composition), not from the threshold. NVE drift is measured on interaction plus kinetic energy.

**Not yet passed.** The 2026-11-07 runs predate the x64 conversion and the interaction-energy extraction. The new gates must be rerun before they are recorded as passed.

**Rerun 2026-11-08: all three gates pass** (`GH200_feasibility_check_2026-11-07.md` §J). Forces 3.0e-5 eV/Å and interaction energies 4.2e-6 eV/atom at float32 in every cuEquivariance combination; float64 totals 2.7e-7 eV/atom from the x64 bundle, cueq off and on, with float64 math dtype shown in both stacks. Two findings: the float64 residual is systematic (about −1.7e-7 eV/atom, all in the interaction energy, E0 part below 2e-10 eV), so not rounding, but within the gate; and mace-jax's `{"backend": "cueq"}` bundle runs the same plain-JAX code as the default bundle. The CUDA kernel needs `"conv_fusion": true` as well, and then covers only the convolution (it passes all three gates; one batch-100 force call drops from 9.2 to 7.1 ms). Plan §4.4 now defines the JAX "cueq on" column that way, which the committee should confirm.


## Addendum 2026-11-10: JAX "cuEquivariance on", kernel evidence and MPS

The implementer asked the committee (Fable 5.1 and GPT-6-Astra, two rounds plus a confirmation round by Astra; Opus 5.5 non-voting) to confirm the plan §4.4 definition of JAX "cuEquivariance on" and the §4.3 MPS-daemon sentence, both written from the measurements in `GH200_feasibility_check_2026-11-07.md` §J and §K.

**Confirmed.**
- JAX "on" is `{"backend": "cueq", "conv_fusion": true}` on mace-jax `e8a6f2d`, labelled "convolution only". `{"backend": "cueq"}` alone runs the same cuequivariance-jax `naive` path as the default bundle. mace-jax `357931b` (a legacy-checkpoint conversion fix) and `dev` change no adapter, so container A stays pinned; any upgrade repeats the conversion checks, the gates and the kernel verification.
- The MPS daemon runs inside container B for the tested forward-compatibility setup (NGC 26.10, compat libcuda 615.71, host driver 595.45). NVIDIA does not document this configuration; revalidate after image or driver changes.

**Corrected.**
- The request said PyTorch `enable_cueq` accelerates "all four layer types" (88 cuEquivariance modules). MACE 0.3.17's wrapper builds Linear and FCTP with `method="naive"`, so only the convolution and the symmetric contraction are eligible for kernels. Module counts are not kernel evidence.
- The fused mace-jax path falls back to `naive` silently on error, and the `cuex` logger counts a call before it returns, so a logged `uniform_1d` call does not prove the kernel ran.
- `x64_verify.py` checks that each source value occurs somewhere in the bundle; it does not establish that 52 tensors match bit-for-bit.
- The MPS pool's timing includes equilibration (`md_torch.py`), unlike every other mode. Each torch worker opens two MPS contexts, so "P ≤ 48" is not a safe bound.

**Adopted (plan §4.2, §4.3, §4.4 cuEquivariance row and CSV columns, §4.5).**
- The decision compares the fastest validated configuration of each stack (mode × cuEquivariance setting, gates passed, execution profile on record); the 2 × 2 grid is descriptive. The caveat states the verified kernel coverage of each "on" setting.
- One untimed execution profile per stack, configuration, batch shape and precision; every timed run records its resolved methods in a manifest, separate from the requested fusion setting. A fallback run is reported as a fallback, not as "on" or "off".
- The float64 residual passes but is unexplained. Timing may proceed provisionally; a tensor-wise conversion check and a layer-wise residual trace are required before the recommendation, and affected runs are repeated if the timed artifact changes.
- MPS: re-measure capacity with the daemon inside B (peak contexts and distinct PIDs), a configurable 300 s start-up deadline with bounded cleanup, and production-only pool timing.
- Script fixes for the implementer: `eq_gates.py` requires every bundle and exits non-zero on a failure; `md_torch.py`'s docstring says the daemon runs inside the image; the torch CSV records the fusion setting and per-module methods.

**Disputes resolved.** Fable held that the residual cannot change either stack's work and need not precede timing; Astra wanted it settled before timing. They agreed on provisional timing in parallel, with a repeat if the fix changes the timed artifact. A fallback run is reported separately (Astra), not counted as "off" (Fable).

**After the 2026-11-10 timing grid** (`reports/t0_timing_summary.md`): the MPS pool is at least 4.3× slower than TorchSim at batch 100, so the MPS items no longer affect the decision. Fix the harness, mark the existing MPS rows "includes equilibration", and do not rerun them.
