# Committee review: charge-state MLFF proposal

Review of `docs/plan/charged_mlff_proposal.md` (version 1, 2026-11-12, branch `charged-mlff`, commit `c1679d1`).
Committee: Fable 5.1 (`claude-fable-5-1`) and GPT-6-Astra (`gpt-6-astra`), both at high reasoning effort, run
through Paseo on 2026-11-12. Opus 5.5 ran the review and merged the text, and did not vote.

Process: four rounds.
1. Each member wrote an independent review.
2. Each member answered the other's review point by point.
3. Both members signed a merged draft, with amendments.
4. Each member accepted the other's amendments.

Both members signed the text below without reservation. Both checked their claims against the proposal, its companion
documents (`charged_mlff_plan.md`, `charged_tdep_validation.md`, `cluster_migration.md`), the code in `d1/`, the
papertrail corpus and the web. The raw trajectory outputs were not available on this machine, so the *(measured)*
numbers were checked for consistency, not reproduced.

---

## Verdict

**Major revision of the document; fund with changes, staged.**
- Authorise Phase 0 now.
- Release seed production (Phase 1) once the sampler fix and the electronic-state QA are in place.
- Train on the 116 seed pairs within the pilot (Phase 2).
- Decide at a gate after Phase 2 on large-scale active learning, production spectra (Phases 3–4) and the
  anion/dication extension. They proceed only if the diagnostic-impact gate (B4) and the O-D reference gate (B3) pass.

## Blocking findings (each with its release point)

**B1. Intensity estimator and cascade coupling (§3.3, §5.8).**
- **Problem.** Take harmonic oscillators with linear dipoles. `d1/mlmd_spectra.py` with `qm=0` (the setting
  `d1/tdep_compare.py` uses) weights integrated strength by (1−e^−x)/x, x = ħω/kT, relative to the
  harmonic-corrected estimator.
- **Size.** This alone makes the 1613/893 cm⁻¹ intensity ratio grow by about 28 % from 300 to 1200 K, with no
  anharmonicity at all. How much of the reported CH-stretch-share rise it explains must be quantified by
  reanalysis.
- **Where it cancels.** The factor cancels in matched-frequency cation/neutral absolute-intensity ratios at the same
  temperature. It generally does not cancel in normalised band shares.
- **Positions and widths.** Positions of sharp lines are unchanged. Centroids and widths of broad bands can shift
  under the frequency-dependent weighting.
- **Fix.**
  - Validate the estimator on harmonic oscillators and on DFT harmonic intensities.
  - Choose the convention and document it.
  - Define the absolute normalisation, and how σ(ν,T) enters the cascade, so that thermal population is counted only
    once.
- **Release point:** before any claim about intensity trends and before any spectrum is released. It does not block
  DFT labelling.

**B2. The sampler censors its target distribution (§5.4, `d1/mlff_sample.py:65`).**
- **Problem.** The sampler drops any frame whose bond graph, at 1.2 × the covalent-radius sum (C–H cut 1.284 Å),
  differs from the input. That is 22 % of frames at 1500 K and 7 % at 1200 K. Their geometries are discarded; only
  the rejection counts are kept. Whether they are transient stretches or real chemistry is not established.
- **Fix.**
  - Keep every candidate frame, labelled by its largest bond elongation and by whether the change persists across
    neighbouring frames.
  - Run DFT, with stability tests, on a stratified sample of the rejected geometries.
  - Add the undisplaced minimum, displacements along normal modes, and controlled boundary geometries (stretches,
    out-of-plane distortions).
  - Do not add dissociating geometries to training automatically without electronic-state checks.
- **Release point:** before Phase 1 seed production. The fix costs hours.

**B3. The O-D deliverable has no accuracy gate (§5.7; G7 is "reported, not gated").** Make G7 a gate that separates
four error sources:
- **Dipole-model error:** DFT dipoles every 2 fs along the model's own trajectories. About 600 core-h for the
  naphthalene cation.
- **PES/MLFF error:** classical Born–Oppenheimer MD on at least one small neutral/cation pair, e.g. naphthalene, at two
  or three temperatures. About 4–5k core-h per temperature pair, before equilibration and overhead. Classical AIMD
  measures the ML approximation error at the chosen level of theory. It does not validate classical nuclear dynamics.
- **Classical-nuclear error:** a small quantum anharmonic benchmark on 2–3 species, e.g. a VPT2-based cascade as in
  Mackie et al. 2018, propagated to the diagnostic. hc ν̃/k is about 2320 K at 6.2 µm and 1285 K at 11.2 µm, so both
  bands are quantum-sensitive over much of 300–1200 K.
- **Sampling uncertainty.**

Require both an uncertainty limit and an accuracy tolerance, not "within the replica scatter", which rewards noisy
references. **Release point:** before expansion beyond the pilot.

**B4. The diagnostic benefit is unquantified (§§1–2, §9).**
- **The gate.** One diagnostic-impact gate in Phase 0, using the corrected estimator. Run the cascade three ways:
  1. fixed harmonic strengths;
  2. transferred neutral trends;
  3. plausible charge-dependent trends, later replaced by calibrated pilot results.
- **What to measure.** The bias and uncertainty in the recovered ionisation fraction, across size, radiation field
  and measurement convention, with an error tolerance set in advance.
- **How to judge it.** Being smaller than the decomposition systematic is not by itself grounds for dismissal. The
  question is whether the correction materially improves the inference.
- **Release point:** before large-scale active learning and production. The gate controls expansion, not whether the
  pilot exists.

## Major findings

**M1. Corrections to the §3 evidence.**
- **Cold-failure count is internally inconsistent.** The text says "six species change in every replica" but reports
  16/36 changed at 50 K. 16 = 4 species × 4 replicas fits the four species named at 50 K. Publish the per-species,
  per-replica table from the raw results. The evidence that *some* species fail, such as the ring opening to 4.5 Å,
  stands.
- **§3.1 omissions.**
  - The port does not reproduce the published intensities: the CH-stretch share is 0.142 published against a replica
    median of 0.230 (`cluster_migration.md` step 2b).
  - The paper names a Nosé–Hoover thermostat, but the released code uses Langevin.
- **§3.3 needs reanalysis:**
  - filter out trajectories that failed the integrity check in `load_mlmd()`;
  - the "width" is an intensity-weighted standard deviation within a moving peak window of fixed ±20 cm⁻¹ half-width,
    not an FWHM. Window truncation limits meaningful broadening measurements: a uniform profile gives 11.55 cm⁻¹,
    though that is not a mathematical upper bound;
  - nominal and actual temperatures are used inconsistently in the prefactor, the selection and the regression;
  - the C–C stretch comparison is cross-method (xTB cation against MLMD neutral), and xTB fails the neutral control in
    that region.
- **Downgrade §3.3** to "preliminary similarity of shifts and widths for selected matched (mostly out-of-plane) bands
  of coronene and pyrene in xTB".

**M2. Electronic-state QA (§5.2, G0–G1).**
- ⟨S²⟩ > 0.80 is a review trigger. ⟨S²⟩ ≤ 0.80 does not certify that the electronic state is adequate. The pilot
  already gives 0.812 at C90 and 1.04 at C144.
- Add these checks:
  - continuity of energy, ⟨S²⟩ and dipole across frames, per species;
  - stability tests on samples, including hot frames and closed-shell singlets;
  - multiple initial guesses;
  - mixed-guess unrestricted calculations for singlet dications;
  - ⟨S²⟩ after annihilation;
  - fixed integration grid (UltraFine) and fixed convergence settings;
  - a ROKS comparison on 2–3 cases only.
- Quarantine problem regions; never drop them silently.

**M3. Domain coverage and test design (§§5.3–5.7).**
- Use dipole and dipole-derivative disagreement in acquisition, not only force disagreement.
- Keep acquisition, validation and an untouched final-test set separate, grouped by skeleton and stratified by size
  bin.
- Add laboratory-anchor species the models have never seen. The named benchmark molecules are currently forced into
  training by `mlff_species.py`.
- Put 60–100 C transfer tests into the seed set rather than "later".
- Make generating the extra charge-state geometries, and assigning their electronic states, a named work package
  wherever PAHdb lacks the required matched structures.

**M4. Alternatives and benchmarks (§2.2, §8).**
- **Phase 0 foundation-model benchmark (one week).** Test charge-aware foundation models as sampler, as fine-tuning
  base and as a cation-versus-neutral pre-check: MACE-OMOL (charge and spin embeddings since mace-torch 0.3.14), UMA,
  and AIMNet2/NSE. They are credible candidates; whether they do better on hot-cation dipoles is unproven.
- **Competing deliverable.** Define the calibrated-correction panel (VPT2 cascades and/or molecule-specific MLMD over a
  small neutral/cation benchmark set) as an explicit competing science deliverable, decided at the gate after
  Phase 2. Foundation models are implementation alternatives; the correction panel is a scientific alternative.
- **Precedent and anchors to add:**
  - Tang et al. 2025 (arXiv:2504.11898): MLMD of pyrene and superhydrogenated pyrene cations, compared with IRMPD.
    Add it as precedent and benchmark.
  - Kim, Wagner & Saykally 2001 (PRL 86, 5691): hot pyrene-cation emission. Add it as an anchor; Tielens 2008 notes
    that the emitters' internal energy is poorly quantified.

**M5. Resources and schedule (§6).**
- **CPU.** This work needs 85–220k core-h against a parent D1 allocation of 15–25k core-h. Request it explicitly.
- **Point count.** 5,800 + 3–5 × 10k = 35.8–55.8k, not 50–80k. Reconcile the core-hours.
- **Phase 0 timing set:** about 150–200 core-h, not under 100.
- **GPU.** State the NVT-length assumption behind 1,700 vs 2,600 GPU-h (20 vs 200 ps).
- **Missing costs:** reference calculations (B3), ensemble training, retraining, state QA and geometry generation.
- **Density fitting.** Retest with a proper JK-fit auxiliary basis before rejecting it; the single test used the
  default auxiliary basis. If Gaussian proves slow, ORCA with RIJCOSX is the fallback.
- **Hangang.** `cluster_migration.md:21` states that arrays of more than 5 tasks are rejected, but the run-book submits
  large arrays. Confirm the CPU QoS in Phase 0, and restructure if needed (e.g. one job per node running concurrent
  inputs).
- **Staffing.** The 3–4 month schedule holds only with a named full-time person.

**M6. Gate statistics and conventions (§5.7).**
- **G4.** Report counts with confidence bounds: 0 failures in 92 runs gives ≤ 3.2 % at 95 %, and less with correlated
  replicas. Make the integrity criterion persistence-based.
- **G2.** Add a per-temperature breakdown and tail errors. Replace dipole MAE as the intensity gate with tests on
  dipole derivatives and band strengths.
- **Frequency convention.** ν_b(T) = ν_PAHdb,scaled + [ν_MD(T) − ν_harm,model], anchored on the model's own Hessian,
  not on a 50 K MD run. Keep the unscaled outputs. Verify the per-mode PAHdb scale metadata: the XML stores 0.9794,
  while Ricca et al. 2025 §2.1 gives three factors.
- **Temperature coverage.** Derive it per size bin from the cascade and the density of states, including the cold
  cooling tail. Tielens's T_max ≈ 2000 (E/N_C)^0.4 K suggests that species under 30 C exceed 1200 K.

## Minor findings

- **Anion basis:** treat 6-31G\* as the PAHdb-compatible baseline. Require diffuse-basis checks of binding, forces and
  dipole derivatives before deciding the basis for anion production, without a fixed ring-count threshold.
- **Dispersion and range separation:** do not change the baseline dataset pre-emptively. Test both on a small
  representative panel and use the results to decide whether changes or systematic-error allowances are needed.
- **G0:** the cross-code energy bound of 1e-5 Ha is fragile. Specify identical numerical settings, and weight forces,
  dipoles and frequencies instead.
- **`sample_T=0`** mislabels frames generated by displacement. Keep the source temperature separately.
- **Hold-out:** stratify it by size bin.
- **Citation:** cite Ricca et al. unambiguously (ApJS 282, 7, 2026).
- **Mai et al.'s test set:** the paper is ambiguous about its 3,583 items. §2.2 says "configurations for 3583 PAHs that
  were excluded from the training set", while §2.3 and the Fig. 2 caption say "conformations" and "configurations". Do
  not describe the Table 1 errors as errors on held-out species. The proposal's G2 gate on held-out species is
  stricter than what Mai et al. demonstrably reported.

## Agreed sound (need not be re-argued)

- B3LYP/6-31G\* as the baseline consistent with PAHdb v4.00 (Ricca et al. 2025 §2.1).
- The PAHdb v4 and Mai et al. 2025 facts (§§2.1–2.4, 4; Table 1) are quoted correctly.
- The pointwise port comparisons rule out a gross porting error.
- The design choices: matched neutral/cation pairs, species-level splitting, active learning, dipoles about the centre
  of mass with a translation test, NVT equilibration followed by NVE collection.
- The Gaussian route, NoSymm, and the unit constants in `d1/g16_io.py`.
- The overall order of magnitude of the cost.

## How the members' positions moved

| Topic | Fable, round 1 | Astra, round 1 | Resolution |
|---|---|---|---|
| Verdict | Fund with changes, staged | Major revision, bounded pilot | Same decision; merged wording above |
| Intensity estimator | Not found | Blocking: ~28 % artefact | Fable verified it. Astra narrowed the scope: it cancels in matched cation/neutral ratios, not in band shares |
| Cold-failure count | Judged consistent across documents | Arithmetic contradiction | Contradiction confirmed within each document; raw table required |
| Sampler filter | Blocking: keep stretched frames | Censoring, add boundary sampling | Merged into B2; Astra corrected the C–H RMS amplitude to ~0.065 Å and rejected "certainly transient" |
| Strongest alternative | Charge-aware foundation models | Calibrated correction panel | Complementary: models in Phase 0, panel as a competing deliverable |
| O-D reference | AIMD at 900 K | Quantum benchmarks | AIMD at 2–3 temperatures for ML error, plus a small quantum benchmark for the classical error |
| Hangang limits | Major | Partly unsupported (CPU QoS unknown) | Confirm the QoS in Phase 0 and restructure if needed |
| Anion basis | Consistent with PAHdb; flag < 3 rings | Separate basis decision | Diffuse-basis check before anion production; no ring threshold |
