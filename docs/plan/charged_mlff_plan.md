# Charged-state MLFF: Gaussian training data, training, active learning (draft, 2026-11-12)

Status: **tools built and Phase 0 inputs ready; nothing submitted on Hangang.** Branch `charged-mlff`.
**Update 2026-11-14: the DFT is GPU4PySCF on the GH200, not Gaussian on Hangang.** Gaussian is not available there
(`docs/research/pyscf_gpu_jax.md`); GPU4PySCF reproduces PAHdb's Gaussian B3LYP/6-31G\* to ≤ 2 µHa with `b3lypg`, 6D d
and the (99, 590) grid, and runs on the GH200's H100 (container D, `containers/D_gpu4pyscf.md`). Seed set ≈ 87
H100-hours with the stability sample, ≈ 33 h on the node's two GPUs (pyscf_gpu_jax.md §5). The Gaussian sections
below (Phase 0, `g16_io.py write`, Slurm) are kept for a spot check if Gaussian ever becomes available.
Reviewer-facing version (self-contained, with evidence, gates, budget and questions):
`docs/plan/charged_mlff_proposal.md`. This file is the run-book.
Context: `docs/plan/charged_tdep_validation.md` (the xTB test: band positions transfer from neutrals to cations,
intensities unsettled; the DFT timing pilot) and `docs/plan/cluster_migration.md` (neutral MLMD, Hangang).

## Goal

Force fields (and dipole models) for PAH neutrals and cations, later anions and dications, that stay stable and
accurate from 50 to ~1200 K (1,500 K for 21–40 C, decision 2026-11-16 in the status log), to compute temperature-dependent spectra per charge state. Mai et al.'s model is
neutral-only, trained on 300 K geometries, and breaks above ~600 K (and for some species at any temperature).

## Facts this plan rests on

- ~~Hangang has Gaussian `gaussian/g16.c02` (module, commercial tree).~~ Not usable (2026-11-13); PAHdb is the only
  Gaussian reference. `normal_cpu`: 680 nodes, 174,080 CPUs (256 per
  node), ~770 GB per node, 24 h, nodes shared between jobs (`Exclusive=NO`); `long_cpu` 2.5 days. About half the
  `normal_cpu` nodes were idle when checked (2026-11-12). Per-user job limits (QoS `normal_cpu`) not yet known.
- **PAHdb v4.00 C/H species are B3LYP/6-31G\***, not 4-31G: 2,524 closed-shell neutrals, 1,469 doublet cations,
  688 anions, ~2,400 dications; only a handful at 4-31G. Mai et al. trained at B3LYP/4-31G (Mai 2025 §2.1).
  PAHdb route: `b3lyp/6-31g* freq ... scf=(xqc,maxconventional=64 maxcycles=128)`; frequencies stored with a
  `scale="0.9794"` attribute.
- mace-jax 0.2.0 (container A): no total-charge or spin input; multi-head fine-tuning; dipole readout blocks and
  losses but no dipole model class. Mai's NeuralIL/EPNN training code is in their release (code Apache 2.0); their
  training data are not.
- DFT cost (PySCF pilot, B3LYP/4-31G, `charged_tdep_validation.md`): ∝ (basis functions)^2.6, cations 1.4–1.7×
  neutrals; 6-31G\* has 15 instead of 9 functions on C, so ~3.4× the 4-31G cost. Gaussian's speed is Phase 0's job.

## Level of theory: B3LYP/6-31G\* (decided 2026-11-12)

| | B3LYP/6-31G\* | B3LYP/4-31G |
| --- | --- | --- |
| Consistent with | PAHdb v4 (harmonic library, geometries, all charge states) | Mai et al.'s model and published spectra |
| Seed-set cost (7,424 points, PySCF-based estimate) | ~13k core-h | ~4k core-h |
| Full data (50–80k points, ≤ 100 C) | ~85–220k core-h | ~25–65k core-h |

Decided 2026-11-12: **6-31G\***. Mai's data are not available, so the neutrals are retrained either way. Reason: so the
anharmonic trends and the harmonic library come from the same electronic structure; at Hangang's scale the extra
cost is days, not weeks. Keep the frequency scaling question separate (PAHdb's 0.9794 vs Mai's 0.9578 for 4-31G).

## Phases

**Phase 0: Gaussian on Hangang (inputs ready; ~150–200 core-h). Will not run (no Gaussian); replaced by the PySCF/PAHdb
match and the GPU/CPU checks of `docs/research/pyscf_gpu_jax.md` §1, §2.2 and §5.2.** `d1/g16_phase0.py` →
`/var/tmp/$USER-pah/d1/g16_phase0/` (30 inputs, `expected.json`; packed as `/var/tmp/$USER-pah/d1/g16_phase0.tgz`,
regenerated 2026-11-12 with the production settings `Int=UltraFine`, `Conver=8`):
- `timing/`: the PySCF pilot's 12 geometries (C10–C144, neutral and cation, same displacement) at 4-31G and 6-31G\*.
  4-31G energies must match PySCF `b3lypg` (expect ≤ ~1e-5 Ha: grids differ); gives Gaussian's real cost per size.
- `origin/`: the naphthalene cation and the same geometry shifted by +3 Å in x. If Gaussian's dipole is about the
  coordinate origin, the two differ by 3 e·Å = 14.41 D in x; then `g16_io.py parse --origin coords` (default) is
  right; if they agree, use `--origin nuclear_charge`.
- `pahdb/`: naphthalene neutral (uid 330) and cation (uid 359) at their PAHdb geometries, 6-31G\* Freq: energies
  against PAHdb `total_e` (−385.89270636 / −385.61361154 Ha) and frequencies against PAHdb's stored values.

- `stability/`: naphthalene neutral and cation as two-step stability → Force jobs, the form production uses for
  flagged frames (a real log for the parser's stability and second-step parsing).

On Hangang (copy `g16_phase0.tgz` to `/appsdata/$USER/d1/` and unpack it there, then):
```bash
cd /appsdata/$USER/d1/g16_phase0
for n in 8 32; do grep -l "nprocshared=$n" $PWD/*/*.gjf > list$n.txt; done
sbatch --array=0-21 --cpus-per-task=8  --mem=20G --time=04:00:00 /appsdata/$USER/pah_stage0/d1/slurm_g16.sbatch list8.txt 1
sbatch --array=0-7  --cpus-per-task=32 --mem=70G --time=12:00:00 /appsdata/$USER/pah_stage0/d1/slurm_g16.sbatch list32.txt 1
```
Check the `--comment` application code first (`showappl | grep -i gauss`; the script assumes `appl=gaussian`).

**Phase 1: seed set, neutrals and cations (~13k core-h at 6-31G\*; about a day on 5 nodes).**
- Species: `d1/mlff_species.py` → `/var/tmp/$USER-pah/d1/mlff_seed/species.csv`: 116 closed-shell neutral /
  doublet cation pairs of the same structure (bond-graph hash), 10–60 C, ~23 per 10-C bin, farthest-point
  sampling in solo/duo/trio/quartet fractions; naphthalene, anthracene, phenanthrene, pyrene, coronene, ovalene and
  circumcoronene always included; 23 pairs held out. Anion and dication uids recorded for later. Small pairs are
  scarce in PAHdb (25 with 10–19 C, 27 with 20–29 C), so nearly all of them are in.
- Geometries (revised after committee review B2, 2026-11-12): `d1/mlff_sample.py`, every frame kept and labelled,
  32 per species and charge: the undisplaced minimum; 3 Gaussian displacements (0.05 Å); 4 normal-mode samples
  (2 at 300 K, 2 at 1000 K, along the GFN2-xTB Hessian's modes); 4 controlled boundary geometries (one C–H stretched
  to 1.25 and 1.40 Å; one atom pushed 0.3–0.6 Å out of plane); 20 GFN2-xTB Langevin MD frames (4 each at
  300/600/900/1200/1500 K, 2 ps equilibration, 0.5 ps apart, FixCom). Labels: `sample_T` ("none" for non-thermal
  frames), `max_stretch`, `max_elong`, `graph_changed`, and for MD `changed_n` (graph changed at the frame, +50 fs,
  +100 fs: 3 = persistent) and `after_change`. 32 × 232 = 7,424 points. The v1 frames (changed frames dropped,
  5,586 kept) are in `frames_v1/`, `gjf_v1/`.
- Stability tests: `g16_io.py write` makes a two-step job (Stable=Opt, then Force from that wavefunction) for frames
  with `graph_changed=1`, all `bound_*` frames and a reproducible random 10 % of frames at ≥ 1200 K; the parser
  records `stability` (stable / reoptimized / not_tested) and ⟨S²⟩ before and after annihilation. All jobs use
  `Int=UltraFine` explicitly.
- DFT (2026-11-14): `d1/dft_seed_run.sh` → `d1/dft_seed_run.py run` (GPU4PySCF, container D, 2 processes per GPU;
  same frame names and stability sample as below, from `gjf/manifest.json`) → `dft_seed_run.py collect` → the same
  extended XYZ as `g16_io.py parse` → `d1/mlff_qa.py`. Pilot of 4 pairs (202 frames) done, all clean. On Hangang (planned for the full set): `d1/slurm_dft_seed.sbatch`, steps in
  `docs/plan/cluster_migration.md` ("Seed-set DFT"). Gaussian route,
  kept for reference: `d1/g16_io.py write` (one `.gjf` per frame, `manifest.json` with frame info) → `d1/slurm_g16.sbatch` →
  `d1/g16_io.py parse` → extended XYZ (`dft_energy` eV, `dft_forces` eV/Å, `dft_dipole` e·Å about the centre of
  mass, `s2`, `scf_cycles`, `config_type`, `sample_T`). Drop frames that did not terminate normally; record ⟨S²⟩.

**Phase 2: train and validate (~100–300 GPU-h).** Same data, two model families: NeuralIL + EPNN per charge state
(Mai's training code; our MLMD driver already runs it) and mace-jax with one head per charge state plus a dipole
model. Ensembles of 3–4 for uncertainty. Gates before scaling: force error on held-out species; no bond-graph changes
at 50–1200 K (`d1/mlmd_integrity.py`); 50 K band positions against PAHdb harmonic; 300 K neutral spectra against
Mai's; cation positions against lab spectra where available; xTB-style temperature trends per charge.

**Phase 3: active learning (~3–5 rounds).** Ensemble MD over many species and temperatures; frames with the largest
force disagreement go to Gaussian; retrain. Widen to 60–100 C, then anions and dications. Stop when integrity and
disagreement stay low on unseen species. Budget at 6-31G\*: of order 100–200k core-h for 50–80k points.

## Risks

- Spin contamination of large cations (PySCF UKS ⟨S²⟩ 0.81 at C90, 1.04 at C144): cap cations at ~100 C or test
  restricted open-shell on a few.
- Open-shell (diradicaloid) neutrals: 9 of 124 seed pairs have an RKS→UKS unstable minimum and are excluded from
  Phase 1 (2026-11-15 entry above). The class is solo-H-rich and grows with size, so it matters most for the large
  PAHs. Every new species (Phase 3, 60–100 C) gets the same minimum test as a gate before labelling; a broken-symmetry
  or multireference treatment for the class is an open decision.
- SCF failures on distorted ions: `SCF=XQC`; failed frames are logged and dropped.
- mace-jax MD speed is not measured yet (T0); it may make production slower than NeuralIL.
- Pure-carbon species (812 in PAHdb v4) are excluded by design.

## Status log

- 2026-11-12: species selection, sampler, Gaussian I/O, Slurm array and Phase 0 inputs built. Seed selection
  re-run after excluding bare-carbon species (812 in PAHdb v4; C24 and C60 had been picked): 655 pairs available,
  116 chosen, 23 held out. Seed geometry sampling running on the project node (64 cores, NUMA node 1):
  `/var/tmp/$USER-pah/d1/mlff_seed/frames/`. The Gaussian parser has been tested only on a hand-written log; Phase 0
  is its first real test.
- 2026-11-12: level of theory decided, B3LYP/6-31G\*. Seed sampling done: 5,586 frames kept of 5,800 (2,791
  neutral, 2,795 cation; dropped 15 % at 1500 K, 7 % at 1200 K, 1.4 % at 900 K). Gaussian inputs written:
  `/var/tmp/$USER-pah/d1/mlff_seed/gjf/` (5,586 `.gjf` + `manifest.json`), packed as `seed_gjf_631gs.tgz` (4.4 MB)
  for copying to Hangang. Not submitted.
- 2026-11-12: committee review (`charged_mlff_committee_review.md`) received. B2 done: sampler keeps and labels all
  frames, adds minimum, normal-mode and boundary geometries; stability two-step jobs for flagged frames; v1 inputs
  withdrawn (do not submit `seed_gjf_631gs_v1.tgz`). Re-sampling running. M1 done: cold-check per-species table in
  `cluster_migration.md`. Batch size K = 10 still holds (a two-step job costs ~1.5× a single point and only ~15 %
  of frames get one).
- 2026-11-12: v2 seed sampling done: 7,424 frames (232 × 32). Changed bond graphs: md 1500 K 146/928 (17 persistent
  over 100 fs), 1200 K 63 (9), 900 K 16 (2), 600 K 5 (4), 300 K 0; nms 1000 K 106/464 (harmonic C–H overstretch);
  bound_ch 232/464 (the 1.40 Å stretches, by design). Persistent changes at 600–900 K all come from one species,
  C13H10_55 neutral (an H moves from C3 to C1 in xTB; probably 1H-phenalene, where H shifts are real chemistry).
  Gaussian inputs written (`gjf/`, B3LYP/6-31G\*, Int=UltraFine): 1,454 two-step stability jobs (all 928 boundary
  frames, 387 MD, 106 nms, 33 disp), 5,970 single Force jobs. Packed as `seed_gjf_631gs_v2.tgz` (5.8 MB). Estimated
  ~14–15k core-h with the stability steps. Not submitted; Phase 0 first.
- 2026-11-12: M2 done (electronic-state QA: `d1/mlff_qa.py`, `d1/g16_stateqa.py`, stability and convergence settings in
  `g16_io.py`). Seed inputs regenerated with them (neutrals: test-only `Stable`, 728; cations `Stable=Opt`, 726; all
  `Conver=8`, `Int=UltraFine`); `seed_gjf_631gs_v2.tgz` (5.9 MB) now holds `gjf/` (7,424) and `gjf_stateqa/` (132).
- 2026-11-12: B1 done (`docs/plan/intensity_convention.md`): MD intensities use qm=3, unclipped, replica NVE T;
  validated on harmonic oscillators (exact at all T) and on xTB naphthalene/coronene ± (MD vs harmonic); default in
  `d1/mlmd_spectra.py` and `d1/tdep_compare.py`; xTB share trends redone; cascade coupling specified.
- 2026-11-12: M3 splits (`d1/mlff_splits.py` → `mlff_seed/species_v2.csv`; v1 kept as `species_v1.csv`). Pairs grouped
  by carbon skeleton (bond-graph hash of the carbons), so hydrogenated variants share their parent's split. The seven
  lab-anchor benchmark species (naphthalene, anthracene, phenanthrene, pyrene, coronene, ovalene, circumcoronene) and
  their skeleton groups are final_test; the rest split ~70/15/15 per 10-C bin. Result: train 76, validation 17,
  final_test 23, plus transfer_test 8 (two pairs per 10-C bin, 61–100 C, skeletons not in the seed set; sampled with
  5 frames each: minimum, one displacement, one MD frame at 300/900/1500 K). Cost of grouping: the 10–19 C bin keeps
  only 11 training pairs of 25 (the anchors' hydrogenated variants follow them into final_test). The 116 seed pairs'
  frames and inputs are unchanged; only their split labels change.
- 2026-11-12: B4 prototype (`d1/b4_impact.py`, `docs/plan/b4_diagnostic_impact.md`): with xTB-sized charge-dependent
  strength trends the recovered cation fraction is biased by 0.03–0.14 (tolerance 0.05); with half that, ≤ 0.037. The
  deciding number is the cation CC-band strength slope; the gate needs a DFT-level value (B3 MD, then the pilot).
- 2026-11-12: B3 part one started (`d1/aimd_pyscf.py`): PySCF B3LYP(VWN3)/6-31G\* MD of naphthalene neutral and cation,
  300 and 900 K, 2 replicas, 2 ps NVT + 20 ps NVE, 8 threads each on cores 0–63, detached
  (`/var/tmp/$USER-pah/d1/aimd/run_all.sh`; checkpoints every 100 fs; rerun to resume). 8.1 s (neutral) and 11.6 s
  (cation) per 0.5 fs step: ~4 and ~6 days; ~10 ps of NVE after ~2–3 days.
- 2026-11-13: transfer_test sampled (16 runs, 80 frames, 2 with a changed bond graph) and written as Gaussian inputs
  (`gjf_transfer/`, 32 CPUs and 60 GB each, 2 stability jobs; ~1k core-h *estimate*). `seed_gjf_631gs_v2.tgz` (6.0 MB)
  now holds `gjf/` (7,424), `gjf_stateqa/` (132), `gjf_transfer/` (80) and `species_v2.csv`. Submit `gjf_transfer` as
  its own array with `--cpus-per-task=32 --mem=70G`.
- 2026-11-13: Gaussian's 6-31G\* uses Cartesian (6D) d functions by default (gaussian.com/basissets: the 6-31G family
  is among the Cartesian-d exceptions); PySCF defaults to spherical (5D). The DFT MD reference had started with 5D:
  stopped after ~1.5 h (outputs in `aimd/spherical_aborted/`) and restarted with `cart=True`, now the default of
  `d1/aimd_pyscf.py` (`--spherical` to override). 4-31G work (timing pilot) is unaffected. Handoff for a PySCF /
  GPU4PySCF / JAX-port study: `docs/plan/handoff_pyscf_gpu_jax.md`.
- 2026-11-14: GPU4PySCF 1.9.2 built for the GH200 (container D) and validated on the H100 against CPU PySCF and
  wasp33a (`docs/research/pyscf_gpu_jax.md` §5). The B3 part one MD checkpoints never hit the Newton-fallback bug; the
  MD itself still has to be finished on another server. Seed-set driver `d1/dft_seed_run.py`; pilot (C10H8_330,
  C24H14_204, C48H20_146, C96H30_3317; 202 frames, 2.1 GPU-h) clean: all converged, no Newton fallback, all 33
  stability tests stable, `mlff_qa.py` quarantined nothing. Results in `/var/tmp/$USER-pah/d1/seed_dft/`. Full run
  (≈ 87 GPU-h, ≈ 33 h on both GPUs) not started.
- 2026-11-15: both D1 DFT jobs move to Hangang (user's decision). Seed-set DFT: `d1/slurm_dft_seed.sbatch`, handoff
  `docs/plan/hangang_seed_dft_handoff.md`. B3 part one MD: continued on GPUs with `d1/aimd_pyscf.py --gpu` (GPU and CPU
  agree to 3e-10 Ha / 8e-6 Ha/bohr at the MD's default grid; NVE energy continuous across the switch), one trajectory
  per GPU, 5.7 / 7.5 s per step (neutral / cation); 2 nodes for ~3.5 days via the self-resubmitting
  `d1/slurm_aimd.sbatch`; handoff `docs/plan/hangang_aimd_handoff.md`. Checkpoints: neutral 3.8 ps, cation 2.5–2.6 ps.
- 2026-11-15: seed set labelled on Hangang (`docs/plan/hangang_seed_dft_handoff.md`): 7,504 frames, 137 GPU-h, all
  converged, 3 Newton fallbacks (cation MD frames, no stability test). Stability tests: 50 neutral frames RKS→UKS
  unstable; 34 cation frames with ⟨S²⟩ > 0.85. Broken-symmetry check (`dft_seed_run.py run --stab-all --bs`: UKS
  from the external-instability orbitals, same settings) on all 124 neutral minima: 9 pairs are open-shell at the
  minimum (E_RKS − E_BS 2.8–22.8 mHa, ⟨S²⟩_BS 0.82–1.12, max |ΔF| 0.35–1.37 eV/Å); 2 are barely unstable (≤ 0.13 mHa,
  ⟨S²⟩ ≤ 0.17, |ΔF| ≤ 0.11 eV/Å); 113 are stable. The gap (0.13 → 2.8 mHa) makes the split threshold-independent.
  **Decision (Phase 1):** pairs whose neutral minimum is RKS→UKS unstable with E_RKS − E_BS ≥ 1 mHa (B3LYP/6-31G\*)
  are excluded as whole pairs, in every split: `d1/seed_exclusions_open_shell.csv` (6 train, 3 transfer_test;
  transfer_test drops from 8 to 5 pairs). The other pairs keep RKS labels on every frame; their unstable frames stay in
  the data, flagged. This is an **open item, not a permanent scope cut**: the excluded pairs are solo-H-rich (median
  solo/H 0.42 against 0.08 for the rest) and large (3 of the 8 pairs at 61–100 C, 6 of 116 below), so the cut acts as
  a solo-H and size selection (the Wang 2026 Fig. 1 bias) and must be stated as such wherever Phase 1 results are
  reported. An open-shell treatment of this class is still to be decided (see Risks). Per-frame and per-minimum
  results: `seed_dft_bs_check.tgz` (`bs_check_summary.csv`, `bs_minima_summary.csv`).
- 2026-11-16: transfer_test backfill started on Hangang without xTB (`d1/transfer_backfill.py`; PAHdb minimum + 4
  Gaussian displacements per charge, `sampler=naive`): 32 candidates screened with the same open-shell criterion,
  7 open-shell (solo-rich again), 16 selected (4 per bin) and labelled (160 frames, all converged); transfer_test is
  now 5 + 16 = 21 pairs (`species_transfer2.csv`). Near-minimum frames only (median RMSD 0.085 Å against 0.51 Å for 1200 K MD),
  so they widen species coverage, not temperature coverage. Results go back with `species_transfer2.csv`. Handoff to
  the project node: `docs/plan/project_node_seed_dft_handoff.md`.
- 2026-11-16 (project node): Hangang results unpacked to `/var/tmp/jjlee-pah/d1/mlff_seed/seed_dft/`; the 202 pilot
  frames are identical to the earlier pilot labels. `mlff_qa.py` on all 7,504 frames: no missing frames and no energy outliers;
  9 dipole outliers (all cations, ⟨S²⟩ 0.76–0.78; 2 in excluded pairs), 196 `s2_review` (134 in excluded pairs), 50
  `unstable` (33 in excluded pairs; kept: C38H16_3164 6, C26H16_814 5, C40H18_10742 5, C34H16_7213 1). Coverage
  (`d1/mlff_coverage.py`) after the exclusions: train 70 pairs (2,240 frames per charge), validation 17, final_test
  23, transfer_test 5 (before the backfill); median solo/H of the kept train pairs 0.08 against 0.50 for the excluded ones; all checks pass.
  **G1 force check passed** (`d1/fd_force_check.py`, 10 frames, both charges, 10–60 C, MD/minimum, the largest-force
  frame at 67.7 eV/Å and the Newton-fallback frame): worst |FD − F_prod| 2.5×10⁻⁵ Ha/bohr (target 1×10⁻⁴); the
  omitted grid-response term ≤ 1.4×10⁻⁵ Ha/bohr; the labels were recomputed on this node to within 1e-12 (Newton
  frame: 4e-10 Ha, 8e-8 Ha/bohr). The 7 kept dipole-outlier cations, stability-tested (`run --stab-all`, internal UKS):
  all stable, energies and dipoles identical to the labels, so the outliers are a DFT–xTB disagreement, not a
  different SCF state. Other states, reached from different guesses, remain for the state-QA side set (item 4).
- 2026-11-16 (project node): transfer2 received (`seed_dft_transfer2.tgz`, hash confirmed), unpacked to
  `/var/tmp/jjlee-pah/d1/mlff_seed/transfer2/`. `mlff_qa.py` on its 160 frames: none missing (the 160 the report lists
  are the 16 unselected candidates), no energy outliers, 7 `s2_review` cations (⟨S²⟩ 0.80–0.85, σ = 0.03–0.05 Å
  displacements), 1 dipole outlier (C78H26_3529_q1_f004, 0.63 e·Å). The dipole outlier and the highest-⟨S²⟩ frame
  (C90H30_171_q1_f004) are stable (`run --stab-all`), with energies and dipoles identical to the labels. The groups are 5 frames, so the
  robust-σ outlier thresholds are loose there (DFT − xTB energy residuals reach 2.3 eV at σ = 0.05 Å and pass); the stability tests carry these
  frames. Coverage with both datasets (`mlff_coverage.py seed_qa.extxyz,transfer2_qa.extxyz
  species_v2.csv,species_transfer2.csv`): transfer_test 21 pairs, 105 frames per charge, 61–100 C (25–30 per 10-C
  bin), median solo/H 0.10 (the 3 excluded pairs: 0.41); no skeleton shared with the seed splits; all checks pass.
- 2026-11-16: **Decision (with the user): MD temperature = the cascade's quantum (microcanonical) temperature; the
  production MD grid goes to 1,500 K for 21–40 C; a hot seed batch at 1,800 K for 21–40 C, in all splits.**
  Basis (`d1/peak_temperature.py`, PAHdb v4 harmonic frequencies, 4,680 pure CₓHᵧ q = 0/+1, cold start): the peak
  temperature (median) is 1,300/1,500/1,860 K for 21–30 C and 1,190/1,380/1,690 K for 31–40 C at 8/10/13.6 eV, and
  has a median under 1,000 K at 10 eV above 50 C. The emission share above 1,200 K (median, hottest decile) at 8 eV is 15 % (32 %)
  for 21–30 C and 0 (8 %) for 31–40 C; above 1,500 K it is ≤ 4 % for all of 21–80 C at ≤ 8 eV (21–30 C at 13.6 eV:
  30 %). So the 1,200 K grid top is the gap for small species, not the 1,500 K training ceiling; 1,800 K is a 1.2×
  margin over the new grid top. Not done: 2,000–2,400 K (only needed at ≥ 10 eV, or below 20 C); species under 20 C
  reach 1,900–2,800 K at 8–13.6 eV and are outside the T0b populations (20–110 C); including them would need a
  photodestruction (H loss) decision. A classical temperature with the same energy is about half the quantum one
  (21–30 C at 10 eV: 880 K against 1,500 K), which is why the rule had to be stated.
  Sampled on the project node (xTB, `mlff_sample.py`): hot batch, 41 pairs × 2 charges, min + NMS 1,800 K ×2 + MD
  1,800 K ×4 (574 frames; changed bond graph in 109/164 NMS and 84/328 MD frames, 13 persistent), every 1,800 K frame
  stability-tested; and the optional transfer2 MD frames, 16 pairs × 2 charges, min + MD 300/900/1,500 K (128 frames,
  5 changed at 1,500 K). DFT on Hangang: `docs/plan/hangang_hot_t2md_handoff.md`, tarball `hot_t2md_1116.tgz`.
- 2026-11-16 (Hangang): hot seed batch and transfer2 MD frames labelled (`docs/plan/hangang_hot_t2md_handoff.md`).
  Check job 1289049 (3 frames against the project node's `smoke_ref/`): worst dE 3.6e-12 Ha, dF 2.4e-12 Ha/bohr,
  dμ 2.5e-11 D, d⟨S²⟩ 1.7e-13, PASS. Production array 1289160 (2 nodes, ~1.9 h): 702 frames, 28.4 GPU-h (mean 146 s),
  none unconverged, no Newton fallback. Hot batch (574): 492 tested, 31 neutral `unstable_rhf_uhf` (RKS label kept,
  flagged) in 10 pairs, mostly the pairs already unstable in the seed set (C38H16_3164 6, C26H16_814 5, C40H18_10742 5,
  C34H16_7213 4) plus C22H14_307 5, C28H14_2226 2 and one each in C34H16_7262, C38H16_7100, C38H16_7102, C40H22_533;
  12/164 NMS and 19/328 MD frames, 13 of them with a changed bond graph; cations: 42 of 287 with ⟨S²⟩ > 0.80, 3 > 0.85
  (max 0.865). transfer2 MD (128): 1 unstable neutral (C84H24_6508), cation ⟨S²⟩ max 0.829. Largest force component
  15.4 eV/Å (an NMS 1,800 K frame). Nothing dropped. Tarball `/appsdata/t641a02/d1/seed_dft_hot_t2md.tgz` (3.2 MB,
  sha256 `92fac185c1c14a17ef677af8d4d64da680f1e19768327e920c59a348f03cc9f7`): `dft/` (702 JSON) and `dft.extxyz`.
- 2026-11-16 (project node): `seed_dft_hot_t2md.tgz` received (hash confirmed), unpacked to
  `/var/tmp/jjlee-pah/d1/mlff_seed/hot_t2md_dft/`. All 114 duplicate `min` frames reproduce the existing labels
  (≤ 2e-8 Ha). `mlff_qa.py` (hot 574, transfer2 MD 128; none missing): the hot batch first gave 8 energy outliers, 7
  of them the `min` frames: in a hot group 6 of 7 frames sit at 1,800 K with a shared DFT − xTB offset of ~1–1.5 eV,
  so the median moves and the reference frame (residual 0 by construction) looks like the outlier. Fixed: the
  reference frame is never flagged as an energy outlier (no effect on the seed set, which had none). Left: 1 energy
  outlier, C22H14_307_q1_hot_f003 (0.94 eV; stable; minao, huckel, atom and 1e guesses all reach the label energy to
  1e-9 Ha, so an xTB error, not another state); 2 dipole outliers, C34H16_6459_q1_hot_f004 (0.53 e·Å, stable) and
  C66H24_3594_q1_md_f003 (0.32 e·Å, stability-tested here: stable, label reproduced); 31 + 1 `unstable` neutral and
  42 + 6 `s2_review` cation frames as Hangang reported. Coverage over all four datasets (`mlff_coverage.py`, output
  `/var/tmp/jjlee-pah/d1/coverage_all.md`): all checks pass; 1,800 K frames (MD + NMS) 180/36/30 per charge in
  train/validation/final_test; transfer_test 21 pairs with MD at 300/900/1,500 K. The `min` row counts each duplicate
  `min` frame again (train 100 against 70 unique); drop the duplicates when building the training data.

## Phase 1 run-book

On the project node (xTB env `/var/tmp/$USER-pah/mamba/xtb`):
```bash
S=/var/tmp/$USER-pah/d1/mlff_seed
python3 d1/mlff_species.py /var/tmp/$USER-pah/data/pahdb-complete-theoretical-v4.00.xml $S   # species.csv, xyz/
# sampling: one process per species and charge (sample_one.sh + xargs -P 64; see $S/sample_jobs.txt)
for f in $S/frames/*.extxyz; do                                                              # Gaussian inputs
  /var/tmp/$USER-pah/mamba/xtb/bin/python d1/g16_io.py write $f $S/gjf --basis "6-31G*" --nproc 8 --mem 16GB
done
```
Copy `$S/gjf` (inputs and `manifest.json`) to Hangang, `/appsdata/$USER/d1/mlff_seed/gjf`, then:
```bash
cd /appsdata/$USER/d1/mlff_seed
ls $PWD/gjf/*.gjf > list.txt; K=10; N=$(( ($(wc -l < list.txt) + K - 1) / K ))
sbatch --array=0-$((N-1))%50 --cpus-per-task=8 --mem=20G /appsdata/$USER/pah_stage0/d1/slurm_g16.sbatch list.txt $K
# after the array: resubmit the same command to retry unfinished inputs (finished ones are skipped)
python3 /appsdata/$USER/pah_stage0/d1/g16_io.py parse gjf seed_dft.extxyz 2> parse_failures.txt   # --origin from Phase 0
```
K = 10 inputs per task keeps every task well under 24 h at 8 cores: a 60 C cation at 6-31G\* is ~6 core-h, ~45 min
wall (*estimate* from the PySCF fit; Phase 0 confirms), so ≤ ~7.5 h per task; 5,586 inputs = 559 tasks. `%50` caps concurrent tasks; adjust to the QoS limits
(`sacctmgr show qos normal_cpu format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU` on Hangang, not yet checked).

## Data layout

| Path (project node: `/var/tmp/$USER-pah/d1/`, Hangang: `/appsdata/$USER/d1/`) | Content |
| --- | --- |
| `mlff_seed/species.csv`, `xyz/` | Selected pairs, PAHdb geometries per charge (`<formula>_<uid>_q<charge>.xyz`) |
| `mlff_seed/frames/*.extxyz` | Sampled geometries, centred on the centre of mass, with `config_type` and `sample_T` |
| `mlff_seed/gjf/*.gjf`, `manifest.json`, `*.log` | Gaussian inputs, frame info keyed by job name, outputs |
| `mlff_seed/seed_dft.extxyz` | Parsed training data: `dft_energy` (eV), `dft_forces` (eV/Å), `dft_dipole` (e·Å, about the centre of mass), `charge`, `mult`, `s2`, `scf_cycles`, `basis`, `source` |
| `g16_phase0/` | Phase 0 inputs, `list8.txt` / `list32.txt`, `expected.json` |

Gaussian logs are compressed after parsing (`xz`), kept on `/appsdata` (not purged), and never committed. Training
data and models stay out of git as well (size); scripts and plans are committed.

## Quality checks after each DFT batch (gate G1 in the proposal; committee review M2)

Nothing is dropped silently: every frame keeps a `qa_flags` label, and quarantined frames stay in the data with the
reason, for a person to decide.

1. Parse: `python d1/g16_io.py parse gjf seed_dft.extxyz 2> parse_failures.txt` (stability result, ⟨S²⟩ before and
   after annihilation, all sampler labels carried over).
2. QA: `python d1/mlff_qa.py seed_dft.extxyz seed_qa --manifest gjf/manifest.json` (xTB env, one core; ~0.1 s per
   frame). Per species and charge it flags:
   - `missing`: inputs with no parsed result (resubmit once, then report);
   - `unstable`: a closed-shell RKS→UKS or an unrepaired internal instability;
   - `s2_review`: doublet ⟨S²⟩ > 0.80 or > group median + 0.05 (a review trigger; ≤ 0.80 certifies nothing);
   - `energy_outlier`: E − E(minimum frame) disagrees with GFN2-xTB's on the same geometries by more than
     max(6 robust σ, 0.5 eV): the SCF likely found another state;
   - `dipole_outlier`: the DFT − xTB dipole difference beyond max(6 robust σ, 0.3 e·Å) of the group.
   Informational: `reoptimized`, `persistent_change`, `graph_changed`, `boundary`. Outputs `seed_qa.extxyz` (all
   frames, flags, xTB energy and dipole), `seed_qa_report.csv`, `seed_qa_missing.txt`.
   Tested 2026-11-12 on PySCF B3LYP/6-31G\* data for the 64 naphthalene seed frames (neutral and cation): no flags on
   clean data; three planted faults (+2 eV energy, ⟨S²⟩ 0.91, an RKS→UKS instability) flagged, nothing else. A wrong
   state within 0.5 eV of the right one passes this check; the state-QA side set (item 4) covers that case.
3. Force consistency on ~10 frames: central finite differences of the energy against the analytic force
   (≤ 1×10⁻⁴ Ha/bohr). `d1/fd_force_check.py` (container D, production settings; per frame the 3 largest and 3
   random components, h = 0.005 bohr, displaced SCFs from the reference density at 1e-11 Ha; also reports the
   grid-response term the production gradient omits). Done 2026-11-16 on 10 frames (status log).
4. State-QA side set (`d1/g16_stateqa.py`): 21 pairs stratified by size (naphthalene, coronene, circumcoronene
   always), minimum and hottest intact MD frame per charge; neutrals re-run with UB3LYP Guess=Mix (broken-symmetry
   check), cations with Guess=Huckel and Guess=Mix, each Stable=Opt then Force; ROB3LYP Force for the three named
   cations. 132 jobs (~300 core-h *estimate*), `make` → `/var/tmp/$USER-pah/d1/mlff_seed/gjf_stateqa/`;
   `report <qa log dir> seed_dft.extxyz` flags species whose default SCF is not the lowest solution
   (by > 1×10⁻⁵ Ha) and gives ROKS−UKS force and dipole differences. Run with or right after Phase 1.
5. Coverage: points per charge, size bin, `config_type`; held-out species must have no training frames.
   `python d1/mlff_coverage.py seed_qa.extxyz species_v2.csv <out prefix>` (kept/excluded after
   `d1/seed_exclusions_open_shell.csv`, solo/H of kept and excluded pairs, qa_flags per split; exit 1 if a pair is
   unknown, lacks a charge, or a skeleton spans two splits).

Settings fixed on every production job: `Int=UltraFine`, `SCF=(XQC,Conver=8,...)`, `NoSymm`. Stability two-step
jobs: open shells `Stable=Opt` (re-optimise), closed shells `Stable` (test only, RKS kept, an instability is
flagged). Mixed-guess singlets for dications come with the dication phase.

## Phase 2 working notes

- Splits: `species_v2.csv` (M3): by carbon skeleton; train for fitting, validation for model selection and active-learning
  stopping, final_test (incl. the lab anchors) and transfer_test (61–100 C) only for the final report. Never split
  frames of one skeleton between sets.
- NeuralIL + EPNN: Mai et al.'s training code under `src/Supplementary_Data_Code/lib/` (`bessel-nn-potentials-velo`,
  `epnn-main`); one model per charge state. Needs a converter from extended XYZ to their input format
  *(not yet written)*. Their EPNN predicts the dipole; for ions check that its charge constraint sums to the total
  charge.
- mace-jax: `mace-jax-train` with `--heads` (one head per charge state) and energy/force keys `dft_energy` /
  `dft_forces`; float32 training, TF32 off as in T0. Dipole: EPNN as above, or a MACE dipole head built from
  `LinearDipoleReadoutBlock` and `DipoleSingleLoss` *(not yet written)*.
- Validation scripts: held-out errors per charge and `sample_T`; integrity runs with the MLMD driver
  (`d1/mlmd_run.py` with a charge-state model); harmonic frequencies from the model Hessian against PAHdb.
  *(The MLMD driver loads Mai et al.'s models; a model-loading hook for our models is needed.)*

## Phase 3 working notes

- Selection per round: ensemble MD (4 models) for ~500 species-charge pairs × 300–1500 K, 20 ps each; frames every
  0.1 ps; keep the top ~2 % by maximum force disagreement per molecule, de-duplicated (at most ~10 per trajectory).
- Each round ~10k Gaussian points; retrain from scratch or fine-tune (to decide by comparing on round 1).
- Stop criteria: on never-trained species, integrity failures < 1 % at ≤ 900 K and < 5 % at 1200 K, and the
  95th percentile of ensemble force disagreement below the G2 force threshold.

## Tools still to write

| Tool | Phase |
| --- | --- |
| Extended XYZ → NeuralIL/EPNN training format | 2 |
| Model-loading hook in `d1/mlmd_run.py` for our trained models | 2 |
| Harmonic frequencies and intensities from a model, against PAHdb (G5) | 2 |
| MACE dipole model or EPNN for ions | 2 |
| Active-learning selector (ensemble MD, disagreement, de-duplication) | 3 |

