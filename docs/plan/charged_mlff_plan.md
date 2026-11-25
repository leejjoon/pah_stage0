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

**G2 energy criterion (decided 2026-11-23, user via the coordinator note, z2mini 21:06).** The proposal's G2
(§5.7) gates forces (MAE ≤ 60 meV/Å) and dipoles; energies are now gated as well, on the **per-pair offset-removed
energy RMSE** of the held-out sets (`mlff_eval.py`, "energy RMSE after per-pair offset"): **≤ 2 meV/atom per charge
and held-out split.** That error governs MD and vibrational spectra (only energy differences within one species and
charge enter). The raw energy RMSE (each species' absolute energy, i.e. the cross-species ionisation-potential-like
offset of the cation head, ~45–49 meV/atom from v2 to v7) is reported but not gated: an absolute IP is a global
property outside a 2 × 5 Å receptive field. Status: v4 0.3–2.0, v7 0.4–1.2 meV/atom (all splits, both seeds): pass.

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
- 2026-11-17: **Decision (with the user): Phase 1 training data drop `unstable` frames only** (49 neutral frames whose
  RKS label lies above a UKS solution; no correct label exists for them yet). Kept, flagged: `s2_review`,
  `dipole_outlier`, `energy_outlier` (every one examined was a correct DFT label, the disagreement on the xTB side),
  `graph_changed` / `persistent_change` (committee review B2). Built with `d1/mlff_dataset.py ... --drop-flags
  unstable` → `/var/tmp/jjlee-pah/d1/dataset_v1/` (`<split>_q<q>.extxyz`, `report.md`): frames q0 / q1, train
  2,387 / 2,420, validation 578 / 580, final_test 753 / 766, transfer_test 152 / 153.
- 2026-11-17 (project node): **electronic-state QA side set done** (Quality checks item 4; `d1/dft_stateqa.py`,
  GPU4PySCF, production settings; 132 jobs on 21 pairs, `/var/tmp/jjlee-pah/d1/stateqa/out`, 0 errors).
  - mix (84; UKS from the alpha HOMO/LUMO rotated 45°, then stability re-optimisation): lower than the label only
    for the positive control C46H20_10574 q0 (already excluded as open-shell): f000 −7.5 mHa, ⟨S²⟩ 1.04, dF max
    0.53 eV/Å; f028 −12.8 mHa, ⟨S²⟩ 1.31, dF max 0.60 eV/Å. So the mixed guess does find a broken-symmetry
    solution where one exists; every other frame is within 2e-9 Ha of its label.
  - huckel (42 cations): none lower. C54H18_37_q1_f000 converged to another doublet 4.0 mHa *above* the label
    (⟨S²⟩ 0.775, dF max 1.08 eV/Å): a separate surface, the label is the lower state.
  - roks (6 cations: C10H8_330, C24H12_18, C54H18_37, f000/f028): ROKS 1.2–1.8 mHa above UKS, dF max 0.026–0.081
    eV/Å (rms ≤ 0.014), dmu ≤ 0.006 e·Å. Using UKS rather than ROKS for the cation labels moves forces by at most
    0.08 eV/Å (no numeric MLFF force target is set yet to compare with).
  - Conclusion: the default SCF finds the lowest state for the kept training labels; no new flags.
- 2026-11-17: **open-shell labels prepared for Hangang** (with the user; `docs/plan/hangang_open_shell_handoff.md`):
  `run --stab-all --bs` on the 207 neutral + 207 cation frames of the 9 excluded pairs and the 49 `unstable` frames
  (463 frames, ~38 GPU-h); new `collect --bs-label` makes a lower converged broken-symmetry solution the label
  (`method=UB3LYP-BS`); `compare` now also checks the BS energy. Tarball `/var/tmp/jjlee-pah/d1/os_relabel_1117.tgz`
  (sha256 `5d6b4fca83a0ceeab33c29b02c048bf9d336ccda6d99a46191944ba41b9d1196`): name lists and 4 reference frames run
  here (C13H10_55_q0_f000 BS −22.79 mHa, ⟨S²⟩ 1.00, as in the 2026-11-15 check; C26H16_814_q0_f008 −0.13 mHa;
  C22H14_307_q0_hot_f001 −1.23 mHa; C34H20_820_q1_f000 stable); RKS/UKS energies equal the existing labels to 2e-12
  Ha. Whether these frames enter training is decided after they return.
- 2026-11-17 (project node): **mace-jax baseline blocked: the 0.2.0 trainer does not update the parameters.**
  Setup committed (`d1/mace_jax_baseline.gin`, `d1/mace_jax_baseline.sh`: preprocess per charge to HDF5, one model
  with heads q0/q1, `/var/tmp/jjlee-pah/d1/mace_v1/`). The HDF5 data match the extxyz exactly (positions, energy,
  forces). Findings:
  - Two-head run (lr 1e-2, 68 epochs, early stop) and lr 1e-3 (8 epochs): force RMSE stays at the RMS force
    (1.6–1.9 eV/Å, i.e. the model predicts ~0 forces), with blow-ups to 20–30 eV/Å and runs of bit-identical epoch
    metrics. Single head: the same.
  - Overfit test, 26 frames of one pair (C11H10_322), `--scaling rms_forces_scaling --clip-grad 10`, 40 epochs: the
    per-batch losses repeat exactly from epoch to epoch (no parameter change). `--debug` (jax_debug_nans) raises
    nothing, so it is not the NaN→0 gradient sanitising in `tools/train.py`.
  - E0s: the model's z_table is built from species indices (0 = H, 1 = C) and E0s are looked up by index, so
    `--E0s '{1: .., 6: ..}'` leaves C at 0 (685 eV/atom error); the gin file keys them by index. `--statistics-file`
    re-imposes the Z-keyed E0s and must not be used. With `--scaling rms_forces_scaling` the reported energy error
    is again ~706 eV/atom, so the E0 handling also interacts with scaling.
  - `amsgrad` (mace-jax's own) fails with the image's optax (`optax._src.utils.cast_tree` missing); `adam` used.
  - Next: check whether the gradients are zero or the update is dropped (one `update_fn` step by hand on the tiny
    set), or train with mace-torch in container B and convert (`mace-jax-from-torch`, x64) instead. NeuralIL
    remains the other route.
- 2026-11-17 (Hangang): **open-shell labels done** (`docs/plan/hangang_open_shell_handoff.md`, `run --stab-all --bs`).
  Check jobs 1294303/1294304 against the project node's 4 reference frames: worst dE 1.8e-12 Ha, dF 6.8e-9 Ha/bohr,
  dμ 1.4e-6 D, dE_bs 3.7e-11 Ha, PASS. Production (4 nodes): seed array 1294319 (431 frames, 41.5 GPU-h) and hot job
  1294320 (32 frames, 3.3 GPU-h); none unconverged. `collect --bs-label`: seed bs 209 / rks 15 / uks 207, hot bs 31 /
  bs_failed 1.
  - 9 excluded pairs, neutrals (207): 192 unstable → BS label, 1.3–27.0 mHa below RKS (median 10.8), ⟨S²⟩ 0.57–1.57.
    15 stable (RKS kept): C13H10_55 f016–f019, f022–f031, exactly its MD frames with a changed bond graph (an H shift
    to a closed-shell isomer; its intact MD frames are all unstable, 17–27 mHa), and **C36H16_7143_q0_f000**: the
    2026-11-15 minima check found it unstable with a BS solution 5.6 mHa lower (same RKS energy to 1e-12); this run's
    external stability test called it stable. So the stability test can miss an instability; its RKS label here is
    wrong and the frame must not be used as `rks`.
  - 49 `unstable` frames of kept pairs: 48 → BS label (seed 17: 0.07–4.5 mHa, ⟨S²⟩ 0.13–0.76; hot 31: 0.01–9.8 mHa,
    ⟨S²⟩ 0.04–1.10). bs_failed: C34H16_7213_q0_hot_f004 (UKS converged 1e-7 Ha below RKS, ⟨S²⟩ 0.005: no real
    broken-symmetry solution; RKS kept).
  - Cations of the 9 pairs (207, all stability-tested): 3 re-optimised to a lower doublet, i.e. a state change from the
    existing labels (C46H20_10574_q1_f002 −1.83 mHa, ⟨S²⟩ 1.54; C58H20_3182_q1_f030 −1.18 mHa, 1.23;
    C60H20_10225_q1_f003 −0.34 mHa, 1.00); 35 with ⟨S²⟩ > 0.85. Newton fallback: C13H10_55_q1_f025,
    C58H20_3182_q1_f029, C60H20_10225_q0_f002. Largest force component 23.1 eV/Å (C88H24_9839_q0_f001, disp).
  Tarball `/appsdata/t641a02/d1/seed_dft_open_shell.tgz` (2.7 MB, sha256
  `d9eb252b83bfa908633da9d923f6b756257b16a9953580a783ed9ec6a0936c0a`): `dft_seed/`, `dft_seed.extxyz`, `dft_hot/`,
  `dft_hot.extxyz`.
- 2026-11-17 (project node): **dataset v2 built** (decision with the user: take the open-shell labels of the entry
  above, no pair exclusions). Tarball received by tailcat, hash matches. All 463 frames match exactly one frame of
  `seed_qa` / `hot_qa` / `t2md_qa` by name, with identical species and positions. Against the existing labels: the 15
  `rks` frames and 204 of 207 cations are unchanged (to 1e-8); the 3 cation state changes move the largest force
  component by 0.16–0.49 eV/Å. New `d1/mlff_relabel.py` swaps the labels in; `mlff_qa.py --reuse-xtb` (new: stored
  xTB values) then recomputes qa_flags, and reproduces the v1 QA files byte for byte when run on them unchanged.
  - Flags: `unstable` cleared on the 81 BS-relabelled frames that had it; set on C36H16_7143_q0_f000
    (`--missed-instability`, so it is dropped). C34H16_7213_q0_hot_f004 (`bs_failed`) keeps `unstable` and is dropped
    as in v1, although its RKS label is within 1e-7 Ha of the UKS solution. New `reoptimized` on the 3 cation state
    changes; one new `dipole_outlier` (C46H20_10574_q1_f002), kept like the other non-`unstable` flags.
  - `mlff_dataset.py` (inputs in v1 order, `--exclude /dev/null --drop-flags unstable`) →
    `/var/tmp/jjlee-pah/d1/dataset_v2/`: train 2610 / 2612 (q0 / q1), validation 580 / 580, final_test 766 / 766,
    transfer_test 168 / 168. Against v1: train q0 +223 = 192 (6 formerly excluded train pairs) + 33 (formerly
    `unstable`) − 2 (the two frames above); q1 +192; validation q0 +2; final_test q0 +13; transfer_test +16 / +15.
    Labels in v2: 240 `bs` (UB3LYP-BS), 14 `rks` with a stability test, 207 `uks`. With no exclusions, v1's
    solo-H selection bias is gone. Work files in `/var/tmp/jjlee-pah/d1/dataset_v2_work/`.
- 2026-11-17 (project node): **mace-torch run1 (dataset v1, `d1/mace_torch_baseline.yaml`) done**: 300 epochs on
  GPU 1 (2 h 35 min), best checkpoint epoch 295, `/var/tmp/jjlee-pah/d1/mace_torch/run1/`. Final error table, RMSE
  energy (meV/atom) / force (meV/Å): train q0 8.4 / 21.1, q1 28.1 / 33.5; validation q0 13.4 / 33.1, q1 51.3 / 87.3.
  Model re-saved on CPU and converted (`conv/run1-jax.npz`); `d1/mace_parity.py` on 10 validation frames per head:
  max |dE| 5.5e-7 eV/atom, max |dF| 2.6e-6 eV/Å, PASS.
  - Validation errors per config_type: q1 forces 64–126 meV/Å (worst md_900K, md_1200K), q0 25–58. The q1 energy
    error is almost all a per-molecule offset: mean error per pair −2.9 to +5.9 eV, but its spread within a pair is
    only 0.02–0.10 eV (q0: offsets −1.0 to +0.8 eV, spread 0.01–0.04 eV). Small cations come out too low and large
    ones too high. The offset difference (q1 − q0) tracks c·N − IP with c = 0.12 eV/atom (correlation 0.83, 17
    validation pairs; IP from the min frames, 5.4–7.6 eV): the cation head spreads the ionisation energy over the
    atoms. A sum of atomic terms cannot hold a roughly size-independent IP, and fitted q1 E0s (linear in n_C, n_H, as
    in `d1/mace_torch_v2.yaml`) are in the same function class, so they are not expected to remove it. For MD of one
    molecule only the within-pair part matters.
- 2026-11-17 (project node): **mace-torch v2 (dataset v2, `d1/mace_torch_v2.yaml`, fitted q1 E0s H −15.175, C
  −1037.284 eV) done**: 300 epochs, best epoch 291, `/var/tmp/jjlee-pah/d1/mace_torch/v2/`. Error table (energy
  meV/atom / force meV/Å): train q0 7.7 / 24.2, q1 29.7 / 32.6; validation q0 10.0 / 36.1, q1 47.4 / 92.0 (run1:
  13.4 / 33.1, 51.3 / 87.3). JAX conversion parity PASS (2.4e-7 eV/atom, 1.7e-6 eV/Å). On the validation sets (q0
  +2 frames against v1), per-frame force RMSE rose for every config_type: q0 39.3 → 44.3, q1 107.7 → 117.4 meV/Å.
  The extra open-shell (BS-labelled) frames make the force fit harder at this model size. Energy error within a
  pair (per-pair mean removed): q0 0.64 → 0.76, q1 1.56 → 1.62 meV/atom; the rest of the q1 energy error is the
  per-molecule IP offset, unchanged by the fitted E0s as expected. Next run (queued, GPU 1): dataset v2 with
  IP-referenced q1 energies (`d1/mlff_ip_ref.py`, `dataset_v2_ipref/`), baseline config, neutral E0s on both heads.
- 2026-11-17 (project node): **IP-referenced run done** (`/var/tmp/jjlee-pah/d1/mace_torch/v2_ipref/`, 18:00–22:12,
  300 epochs; best checkpoint by summed validation loss at epoch 90, flat after about epoch 100). Error table: train
  q0 7.3 / 32.2, q1 12.1 / 47.2; validation q0 9.4 / 41.1, q1 10.2 / 104.5 (meV/atom / meV/Å). The per-molecule
  offset is gone (rms of per-pair mean q1 energy error 2.07 eV in run1, 2.09 in v2, 0.32 here), but forces got worse,
  on both heads and in training as well (underfitting, not overfitting). Per-frame validation force RMSE q0 / q1:
  run1 39.3 / 107.7, v2 44.3 / 117.4, IP-ref 48.9 / 134.0 meV/Å; energy within a pair q1 1.56 / 1.62 / 1.82
  meV/atom. Even its last epoch (q1 105.6 meV/Å in the training log) is behind v2's best (92.0). With energy weight 1
  against force weight 100 the energy term is a small part of the loss, so the cause is not settled; all three are
  single runs (seed 7) and run-to-run spread is unknown. IP-referenced q1 labels are not adopted for now.
- 2026-11-18 (project node): **v2 seed repeat (`--seed 8`) and larger model (v2, `--num_channels 128 --max_L 2`)
  done** (`mace_torch/v2_seed8/`, 2 h 45 min; `mace_torch/v2_large/`, 3 h 36 min, both 300 epochs). Validation
  energy (meV/atom) / force (meV/Å) from the training log, at the best checkpoint and at the last epoch:

  | run | best epoch | q0 best | q1 best | q0 epoch 299 | q1 epoch 299 |
  |---|---|---|---|---|---|
  | run1 (v1) | 295 | 13.4 / 33.1 | 51.3 / 87.3 | 13.3 / 33.2 | 51.4 / 87.7 |
  | v2 | 291 | 10.0 / 36.0 | 47.4 / 92.0 | 9.8 / 36.0 | 47.6 / 92.2 |
  | v2 seed 8 | 105 | 19.2 / 41.2 | 48.3 / 94.6 | 7.0 / 38.5 | 49.0 / 95.0 |
  | v2 IP-ref | 90 | 9.4 / 41.1 | 10.2 / 104.5 | 8.0 / 39.2 | 8.9 / 105.6 |
  | v2 large | 86 | 14.3 / 34.5 | 45.9 / 92.1 | 7.7 / 30.7 | 47.3 / 92.7 |

  - **Checkpoint selection:** the best checkpoint is chosen by the summed validation loss, which is force-dominated
    (with weights 1 and 100 the loss tracks F²: IP-ref against v2 seed 8, loss ratio 1.30, (105.6/92.2)² = 1.31;
    the q1 energy offset is not the cause, IP-ref has none and stalled too). Its largest part, the q1 force term,
    plateaus and jitters from about epoch 90–120 (minimum q1 force at epoch 119 / 90 / 118 for seed 8, IP-ref,
    large), so in three of five runs selection stopped near epoch 86–105 while q0 forces kept improving to the end.
    Only the selected checkpoint is kept (no epoch-299 weights survive). mace logs "Stopping optimization after N epochs without improvement" (patience 40)
    every epoch after that but trains on to max_num_epochs and then loads the early checkpoint. So the error tables
    of seed 8, IP-ref and large describe undertrained models. Compare the epoch-299 columns.
  - **Seed spread** (v2 against seed 8, epoch 299): q0 force 36.0 / 38.5, q1 92.2 / 95.0 meV/Å, about 3 meV/Å. The
    IP-ref q1 deficit (about 12 meV/Å) is larger than that; v1 against v2 (q1 88 against 92–95) is about at the
    spread.
  - **Larger model:** q0 forces 36 → 30.7 meV/Å (beyond the spread), q1 unchanged (92.7). The q1 force error is
    not a capacity limit.
  - **One validation molecule dominates:** C12H8_385 (a five-membered ring, skeleton not in train) carries 31 % of
    v2's q1 squared force error (349 meV/Å on its own frames) and 23 % of q0's (118 meV/Å); next is C12H10_202
    (no fused rings; 13 % of q1). Without C12H8_385, error-table force RMSE q0 / q1: run1 30.3 / 74.4, v2 32.0 /
    77.2, IP-ref 37.9 / 90.9 meV/Å. Validation error mostly measures transfer to unseen skeletons, as the split
    intends; the q1 train/validation gap (31 against 92) is that transfer gap.
  - Ring sizes (C–C < 1.75 Å, minimum cycle basis of the neutral min frames): five-membered rings in 4 of 76 train
    pairs (C16H10_387, C19H14_349, C20H10_159, C48H20_146), 2 of 17 validation (C12H8_385, C20H12_389), 1 of 23
    final_test, 0 of 24 transfer_test. So C12H8_385 is not the only five-ring case; it is the smallest and most
    strained one, and C20H12_389 (also a five-ring) has an ordinary error.
- 2026-11-18 (project node): **test splits evaluated** (five models on dataset v2's final_test and transfer_test,
  float32, GPU 1; `mace_torch/conv/<run>_test.npy`). Force RMSE (meV/Å) q0 / q1; seed 8, IP-ref and large at their
  early checkpoints (epoch 105 / 90 / 86):

  | run | transfer_test all (24 pairs) | transfer, kept v1 pairs (21) | transfer, formerly excluded (3) | final_test w/o bound_* and two hydro-pyrenes |
  |---|---|---|---|---|
  | run1 (v1) | 55.4 / 73.6 | 35.5 / 70.0 | 144.6 / 102.4 | 46.0 / 119.0 |
  | v2 | 41.2 / 69.6 | 36.6 / 68.3 | 72.2 / 80.8 | 54.1 / 114.2 |
  | v2 seed 8 | 43.5 / 72.9 | 39.3 / 73.0 | 72.8 / 72.3 | 55.5 / 116.5 |
  | v2 large | 47.4 / 67.0 | 44.1 / 66.6 | 72.3 / 70.8 | 48.3 / 116.0 |
  | v2 IP-ref | 42.0 / 67.2 | 37.5 / 66.1 | 73.0 / 77.3 | 54.9 / 128.2 |

  - transfer_test (C72–C96 and the other large pairs, the production sizes): energy within a pair 0.35–0.51
    meV/atom for all. v2's gain over run1 comes from the 3 formerly excluded (open-shell) pairs, whose q0 force
    error halves (145 → 72); on the other pairs the runs are within the seed spread.
  - final_test as a whole (q0 80–147, q1 181–446 meV/Å) is dominated by two hydrogenated pyrenes (whole skeleton
    held out): C16H16_336 (6 sp3 C; train has at most 3) and C16H12_337. On their non-boundary frames the errors
    are 0.28–0.59 eV/Å (q0) and 0.34–1.66 eV/Å (q1); one C16H16_336 q1 bound_oop frame (DFT max |F| 68 eV/Å) alone
    gives 30–41 % of the q1 squared error. sp3 census (neutral min frames, C with 4 neighbours): train 10 of 76 pairs
    (1–3 sp3 C each), validation 2, final_test 6, transfer_test 0. Hydrogenated PAHs beyond about 3 sp3 C are out of
    the training domain.
- 2026-11-18 (project node): **scope decision (user): heavily hydrogenated PAHs are not a science target**; the two
  hydrogenated pyrenes are reported separately from the final_test headline, no DFT is spent on such species, and the
  lightly hydrogenated training pairs stay. Basis: the 3.4 µm aliphatic band is "a minor amount of H bonded to sp3 C"
  (methyl or superhydrogenated), and "the abundance of superhydrogenated PAHs is expected to be very small throughout
  the Orion Bar as such extra H's are readily lost in the strong FUV radiation field" (Chown et al. 2024, PDRs4All IV,
  §4.3); in the ISM "the abundance of aliphatic groups is very small" and the 3.4 µm band disappears toward the Orion
  Bar surface (Tielens 2008, ARA&A, §5.4.3).
  - Correction to the entry above: the two pyrenes fail for different reasons. C16H16_336 has 6 edge CH2 (more sp3 C
    than any training pair). C16H12_337 has only 2 sp3 C, but they are interior CH (15,16-dihydropyrene type, H out of
    the ring plane); every hydrogenated training pair has edge CH2 or CH3 only. 337 is the larger outlier: v2 is off
    by 0.74 eV/Å even at its DFT minimum (q0; q1 1.0), constant over all its frames, i.e. an unlearned motif, not a
    label problem (s2 0.77, all frames qa ok). Interior H is not the expected form of interstellar
    superhydrogenation (H adds at edge carbons, giving CH2), so it falls under the same decision.
  - final_test force RMSE (meV/Å, q0 / q1, w/o bound_*) with only 336 removed: run1 90 / 145, v2 133 / 202, seed 8
    136 / 150, large 80 / 132, IP-ref 112 / 144; with both removed as in the table above. The headline final_test
    number is quoted without both, with this reason.
- 2026-11-18 (project node): **v2 large retrain keeping every checkpoint** (`mace_torch/v2_large_all`, GPU 1, started
  13:30, ~45 s/epoch): `GPU=1 CONFIG=d1/mace_torch_v2.yaml d1/mace_torch_baseline.sh dataset_v2 v2_large_all
  --num_channels 128 --max_L 2 --save_all_checkpoints --keep_checkpoints`. With `--save_all_checkpoints` mace saves
  a checkpoint every evaluation and its end-of-run `load_latest` takes the highest epoch, so `baseline.model` should
  be epoch 299 (to confirm from the final "Loading checkpoint" line); all epochs stay on disk (~22 MB each) for
  choosing by held-out error instead of the force-dominated validation loss.
  - Done 17:18; the final model is epoch 299 (log: "Loading checkpoint ... epoch-299.pt"). Force RMSE (meV/Å) train /
    valid: q0 15.2 / 32.3, q1 19.6 / 94.6 (v2: valid 36.0 / 92.2). The larger model helps q0 a little and q1 not at
    all; q1 fits its training set nearly as well as q0 but generalises 5x worse, which points to cation data
    (amount or diversity), not capacity.
- 2026-11-18 (project node): **G5, frequency half, v2** (`d1/mlff_accept.py freq`, CPU, float64; PAHdb geometry
  relaxed with the model, analytic model Hessian vs PAHdb v4.00 unscaled harmonic frequencies, sorted-order pairing;
  `accept/freq_v2*.json`). Strong bands (PAHdb I > 10 % of max), RMS (cm⁻¹), median and pass (≤ 10) per split,
  without the two pyrenes:

  | split | q0 median, pass | q1 median, pass |
  |---|---|---|
  | validation (17) | 5.5, 15/17 | 5.2, 15/17 |
  | final_test (21) | 5.4, 20/21 | 7.6, 13/21 |
  | transfer_test (24) | 8.4, 13/24 | 7.1, 18/24 |

  Worst: q0 C96H30_3317 16.3; q1 C10H12_332 33.2, C12H10_202 26.5, C54H18_37 25.8. Pyrenes: C16H16_336 18.6 / 15.2,
  C16H12_337 26.3 / 94.8. Five relaxed model minima have one imaginary mode (to inspect). Sorted-order pairing can
  swap near-degenerate modes, so the fail counts are provisional until the per-mode tables are checked. The intensity
  half of G5 needs a dipole model and is not tested.
- 2026-11-18 (project node): **G4 subset, v2** (`d1/mlff_accept.py md`, ASE, float32; 4 held-out pairs C12H8_385,
  C28H14_2226, C72H22_9911, C96H26_9397 × q0/q1 × 300/900/1200 K × 2 replicas; 20 ps Langevin NVT + 20 ps NVE at
  0.5 fs; `accept/md_v2/`). 8 per GPU, ~1.7 h per batch of 8 small molecules; a second worker on GPU 0 (free after
  the user's job ended) takes the list from the end, with lock files in `md()` against double runs. First 16:
  15 intact, NVE drift ≤ 0.5 meV/atom; C12H8_385 q1 1200 K replica 1 lost a C–H bond (C9–H13).
  - All 48 done (2026-11-19 ~01:00; ~1.8 h per batch of 8 whatever the size, i.e. ASE/Python overhead-bound; the
    lock files kept the two workers apart). 47/48 intact: 300 K 16/16, 900 K 16/16, 1200 K 15/16 (94 %; the gate asks
    ≥ 95 % at 1200 K, but one event in 16 runs cannot resolve that). C72H22_9911 and C96H26_9397 (the formerly
    excluded open-shell pairs, production size) intact in all 24 runs. NVE drift over 20 ps |ΔE| ≤ 0.9 meV/atom
    (largest C96H26_9397 q0 300/900 K, −0.8/−0.9). Not the full G4 (50 ps NVE, 4 replicas, more species).
- 2026-11-19 (project node): **follow-ups on the v2 acceptance checks.**
  - **Symmetry-resolved G5** (`d1/mlff_accept.py symfreq`, GPU, float64): the model Hessian is projected onto the
    irreps of the abelian point group (D2h, D2, C2v, C2h, Cs, C2, Ci, C1; axis-aligned operations) and paired with
    PAHdb's mode labels within each irrep. 124/128 entries resolved (4 D6h/D3d/C3 keep sorted pairing); every resolved
    case selects the PAHdb file's own axes (Gaussian standard orientation), which is then used for the 3 ambiguous D2/C2
    cases. PAHdb frequencies are now sorted after unscaling. Sorted pairing distorts the strong-band RMS both ways:
    coronene⁺ 16.8 → 4.7 (its imaginary mode shifted every pair by one), C54H18_37⁺ 25.8 → 7.1, biphenyl⁺ 26.5 → 12.8.
    Strong-band RMS (cm⁻¹) median and pass (≤ 10), without the two pyrenes: q0 validation 5.8 (15/17), final_test 6.2
    (19/21), transfer 9.4 (13/24); q1 5.4 (14/17), 7.1 (15/21), 6.3 (17/24); all 124: median 6.2, 93 pass. Worst:
    C10H12_332⁺ 34.6, C16H12_335⁺ 19.9, C100H26_7494⁺ 18.8, C12H8_385⁺ 17.4, C96H30_3317 16.3. Large neutrals are the
    weakest group. `accept/symfreq_v2*.json` (per-mode tables with irreps).
  - **Imaginary modes**: coronene⁺ (C24H12_18 q1) −326 cm⁻¹ (B3g) is a model artifact: along that mode the model has
    a ~1 meV double well (minimum near 0.01 Å max displacement, then stiff), while DFT (T1 settings, CPU PySCF) rises
    monotonically (+0.15, +1.7, +17 meV at 0.005, 0.01, 0.02), so PAHdb's D2h structure is a DFT minimum. Harmless for
    MD, wrong for the harmonic analysis of that mode; coronene⁺ is Jahn–Teller active (open-shell surface again). The
    analytic model Hessian was checked against finite differences of forces (max deviation 5e-4 eV/Å²). The others
    (C42H24_3973, C78H26_3529, C90H30_171; −2 to −11 cm⁻¹) are flat modes of large molecules.
  - **The G4 "bond loss" is a thermal snapshot**: in the final frame C9–H13 is 1.33 Å, still bound (cut 1.284 Å).
    A DFT C9–H13 stretch scan of C12H8_385 agrees with the model within 7 meV to 1.6 Å (q0 and q1), so the model's C–H
    is not too soft. Reruns at 1200 K saving every 5 fs (`accept/md_v2_traj/`, q0/q1 × 4 seeds): 3–11 % of frames have
    some C–H beyond the cut, each excursion lasting ≤ 10 fs (one vibration), max 1.49 Å, never persistent; q0 and q1
    alike. A single-frame check therefore "fails" a few % of 1200 K runs by chance. mlmd_integrity.py should require a
    change to persist (e.g. present in all frames over the last ~100 fs, or a time-averaged distance) before G4 is
    scored. The flagged original run saved no frames, so its persistence cannot be checked; its final state (1.33 Å,
    still bound) is consistent with such an excursion. The reruns also show C–C graph changes in 0–3 single frames
    each (transient, same cause).
- 2026-11-19 (project node): **q1 force-weight run** (`mace_torch/v2_q1fw3`, GPU 1, started 15:46, ~33 s/epoch):
  v2 config and size, `--save_all_checkpoints --keep_checkpoints`, on `dataset_v2_q1fw3` = dataset_v2 with
  `config_forces_weight=3.0` in every train_q1 header (mace's per-frame force weight; checked with
  `mace.data.utils.config_from_atoms`: q1 forces 3.0, q0 1.0). Validation files unchanged, so validation errors
  compare with v2 directly. Caveat: q1 already fits its training set (19.6 meV/Å in the large run), so up-weighting
  attacks under-fitting, not the generalisation gap; a null result would support more cation data (Phase 3).
  **Result (done 18:34, epoch 299): null to negative.** Force RMSE train/valid meV/Å (energy meV/atom):
  q0 33.1/44.4 (19.4/24.4) vs v2 24.2/36.1 (7.7/10.0); q1 35.9/97.3 (33.4/46.4) vs v2 32.6/92.0 (29.7/47.4).
  Best saved epoch on valid q1 is 55 (96.4), still worse than v2 (92.0). Up-weighting did not even lower the
  q1 training error and cost the q0 head (shared trunk), so it is dropped. With the large run, this points to cation
  data, not model capacity or loss balance, as the limit: the next lever is an active-learning round for cations.
- 2026-11-20 (project node): **active-learning selector** `d1/mlff_al_select.py` (jobs / explore / score / select;
  DFT on Hangang per the user). Explore: ASE Langevin NVT with one model, train-split pairs only (exits on a held-out
  pair), a frame every 50 fs after 1 ps, mlff_sample.py labels plus changed_n, stop at the first persistent graph
  change. Score: committee force disagreement (max over atoms of the per-atom force std). Select: score window
  [lo, hi], greedy per trajectory with a time gap, global budget; packaged as `frames_<tag>/` + `gjf_<tag>/manifest.json`
  (g16_io.py write) with the DFT minimum first, as the hot/t2md batches. Smoke test (`/var/tmp/jjlee-pah/d1/al_smoke`,
  committee v2 + v2_large_all + v2_q1fw3, 2 cation pairs, 1 ps): explore -> score -> select -> dft_seed_run.py run on 2
  frames -> collect all work; held-out C10H8_330 refused. Committee scores 0.11-0.34 eV/A; 1 ps took ~60 s per run.
- 2026-11-20 (project node): **AL round 1 explored, selected, packaged** (`/var/tmp/jjlee-pah/d1/al1/`). 228 runs
  (76 train cations x 300/900/1,200 K, 10 ps, driver v2, 20 workers on both GPUs, 01:20-07:24). 21 stopped at a
  persistent graph change: ~10 single C-H losses (mostly 1,200 K) and ~9 whole-molecule break-ups, all in
  CH2-containing cations (C13H10_54/55 incl. 300 K, C15H12_326/328, C19H14_349, C20H14_344/346, C22H16_356): the
  cation model's weakest region is sp3 sites, as for the hydrogenated pyrenes. Committee score (v2, v2_large_all,
  v2_q1fw3) median 0.045 / 0.080 / 0.100 eV/A at 300 / 900 / 1,200 K (p98 0.13 / 0.25 / 0.30). Selection: score
  0.08-2.0, max_stretch <= 1.5 (new `--max-stretch`: a blown-apart C13H10_55 frame, stretch 9375, scored only 0.24
  because the models agree on near-zero forces, and its SCF ran > 20 min), <= 15 per trajectory 0.3 ps apart:
  2,347 frames + 76 minima, 386 stability-tested; ~48 process-h estimated from seed timings. Reference frames
  labelled locally (C60H20_10225 f035 153 s, <S^2> 0.850; C36H16_7143 f001 33 s; C13H10_54 f028 43 s).
  Tarball `al1_hangang.tgz` (sha256 `7d58f6d7...2836`), handoff `docs/plan/hangang_al1_handoff.md`.
- 2026-11-20 (project node): **ingest pipeline ready for the AL labels.** `d1/mlff_al_ingest.sh` (mlff_qa.py on the
  batch; drop its min frames, which repeat base minima but sit up to 3.5e-5 A off after centring, beyond
  mlff_dataset.py's 1e-6 A duplicate check, and compare their energies with the base instead; mlff_dataset.py with
  v2's inputs + the batch; held-out files checked byte-equal to v2; prints the train/eval commands) and
  `d1/mlff_eval.py` (several models side by side per split/charge: force RMSE, energy RMSE, energy RMSE after a
  per-pair offset, force RMSE per config_type; `--files` for any labelled batch). Dry run on 4 locally labelled al1
  frames: QA, min-drop (dE 0), dataset (train q1 2612 -> 2615, held-out unchanged), 1-epoch mace training and eval
  all work; mlff_eval reproduces mace's own validation table (v2 36.1 / 92.0, large 32.3 / 94.6 meV/A).
  - Finding from mlff_eval on v2: the q1 validation **energy** error is almost entirely a per-pair offset (47.4 ->
    1.6 meV/atom after removing each pair's mean; q0 10.0 -> 0.8). The q1 **force** error is concentrated in a few
    species: C12H8_385 349, C12H10_202 213, C11H10_320 117, C42H22_156 105 meV/A; 70.5 without the top two. It is as
    large at `min` (81) and md_300K (92) as at 1,200 K (98): a per-species error, not a high-temperature one. Round 1
    samples configurations of training species; if the gap is chemical (small cations, five-membered rings, biphenyl
    type), it will narrow little and round 2 should add species (more small cation pairs) rather than frames.
- 2026-11-20 (Hangang): **AL round 1 labelled** (`docs/plan/hangang_al1_handoff.md`). Check job 1315953 (3 frames
  against the project node's `ref/`): worst dE 3.6e-12 Ha, dF 1.7e-12 Ha/bohr, dμ 5.2e-12 D, d⟨S²⟩ 1.1e-13, PASS.
  Production array 1315973 (4 nodes, the user's limit, instead of the handoff's 5; ~1.8 h): 2,423 frames, 52.8 GPU-h
  (mean 78 s), none unconverged; Newton fallback 5 (C20H10_159 f024, C34H20_820 f014, C36H18_3994 f011/f025/f040).
  ⟨S²⟩ > 0.80: 197 (1,200 K 102, 900 K 73, 300 K 20, min 2); > 0.85: 43 (> 0.90: 20, max 1.467), all in the formerly
  excluded open-shell pairs C58H20_3182 17, C60H20_10225 11, C46H20_10574 9, C34H20_820 6. Stability (386 tested):
  4 re-optimised to a lower doublet: **C20H10_159_q1_al1_f032 −6.39 mHa** (⟨S²⟩ 0.769, so a different doublet, not
  spin contamination), C58H20_3182 f024 −0.87 / f025 −0.53 mHa (⟨S²⟩ 1.26/1.25), C34H20_820 f028 −0.09 mHa (0.865).
  Largest force component 14.3 eV/Å (C40H18_10742_q1_al1_f021, 1,200 K). Nothing dropped. Tarball
  `/appsdata/t641a02/d1/al1_dft.tgz` (10.4 MB, sha256 `a6f21e39641f862cdf08920ca8071bc07d06092fcc749f25e764d4f0bf6c5a81`):
  `dft/` (2,423 JSON) and `dft.extxyz`.
- 2026-11-21 (Hangang): **AL round 2 labelled, 1,701 of 1,728 frames** (`docs/plan/hangang_al2_handoff.md`; pulled
  from myhub `pah-data/al2_hangang:2026-11-21`). Check job 1327217 (3 `ref/` frames) PASS. Production array 1327330
  (4 nodes; 34 min to 1.74 h per task, 24.1 GPU-h, mean 51 s): task 1's shards hit their own 1.7 h budget and stopped
  cleanly with **27 frames unrun** (f008-f011 of C15H10_341 q0/q1, C15H14_354 q0/q1, C18H12_2355 q0/q1, C9H8_495
  q0/q1; list in `missing_27.txt`), to be labelled on the project node (also running the full set). Of the 1,701: none
  unconverged, no Newton fallback; doublet ⟨S²⟩ > 0.80: 24 (mostly C19H12_63_q1, max 0.835); stability (304 tested):
  9 `unstable_rhf_uhf` neutrals (C19H12_63_q0 f007-f011/f027/f029, C24H14_2326_q0 f006/f030). Largest force
  components 80.7 (C20H14_358_q1_f010) and 74.0 eV/Å (C17H14_342_q1_f011), both `bound_oop` frames; next 25.2.
  Nothing dropped. Tarball `/appsdata/t641a02/d1/al2_dft.tgz` (4.8 MB, sha256
  `0efad4568f57b525e9327e089f241692be1c6f707353fa8a99803a5225e9fc1c`; `dft/`, `dft.extxyz`, `missing_27.txt`), also
  in myhub as `pah-data/al2_dft:2026-11-21`.
- 2026-11-22 (Hangang): **AL round 3 track 2 labelled, 2,368/2,368 frames** (`pah-data/al3_own:2026-11-22b`, 37
  hand-built pairs, 9–25 C, frames and manifest from the project node). Array 1335773 (4 nodes, `slurm_dft_seed.sbatch`,
  `FRAME_DIRS=frames_own`), 29.3 GPU-h (mean 44.6 s). The 3 `ref/` frames, labelled in production: worst dE 1.8e-12 Ha,
  dF 5.2e-13 Ha/bohr, PASS. None unconverged; Newton fallback 1 (C18H10_x30_q1_f013); doublet ⟨S²⟩ > 0.80: 6
  (C10H10_x02_q1_f007 1.265, others ≤ 0.827); stability (416 tested): 1 re-optimised (C18H14_x40_q1_f030). Largest force
  component 29.8 eV/Å (C16H10_x27_q0_f003, disp). Nothing dropped. Tarball `/appsdata/t641a02/d1/al3_own_dft.tgz`
  (5.7 MB, sha256 `565f17295bff5d90f8af9c19ac10eea370ac3a6c38903752d7728bb0a8e98e62`), myhub `pah-data/al3_own_dft:2026-11-22`.
- 2026-11-22 (Hangang): **AL round 3 track 1 labelled, ranks 1–32 of `al3_track1:2026-11-22`, 2,048/2,048 frames.**
  One node, array 1335774 (`d1/slurm_al3.sbatch`, `MAXRANK=32`), 12:14–19:06; DFT 34.7 GPU-h (mean 61.0 s). Cations
  sampled with `--mult q+1` (the package README's `2q+1` would make them triplets). None unconverged; Newton fallback 2
  (C27H14_73_q1 f003/f023); doublet ⟨S²⟩ > 0.80: 48 (≤ 0.875; C27H14_73_q1 14); stability (421 tested): 32
  `unstable_rhf_uhf` neutrals in 7 species (C32H18_1515_q0 and C32H18_1742_q0 9 each), 2 re-optimised. Largest force
  component 26.1 eV/Å (C28H16_2185_q1_f002, disp). Nothing dropped. Ranks 33–100 not run. Tarball
  `/appsdata/t641a02/d1/al3_hangang_dft.tgz` (9.2 MB, sha256 `419122aa9b46a18fbe45620226119cbc2b9781974ab2550d5f69555d4817a36d`;
  `dft/`, `dft.extxyz`, `frames/`, `gjf/manifest.json`, `minima/`), myhub `pah-data/al3_hangang_dft:2026-11-22`.

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
- **Training data** (`d1/mlff_dataset.py`, 2026-11-17): from the QA'd extxyz files and species tables to one
  extxyz per split and charge, frames copied verbatim. Applies the open-shell exclusions (whole pairs), drops the
  repeated `min` frames of later batches (centred positions checked to 1e-6 Å), and drops frames by `qa_flags` only
  if `--drop-flags` names them (the choice is the user's; the report states it). `report.md` gives frame counts per
  split and charge after each step and the solo-H bias sentence. Seed + transfer2 + hot + transfer2 MD, no flags
  dropped: train 2,420, validation 580, final_test 766, transfer_test 153 frames per charge.
- Isolated-atom energies (UKS b3lypg/6-31G\* cart, (99, 590), CPU PySCF): H doublet −0.5002727827 Ha
  (−13.613116 eV), C triplet −37.8462799650 Ha (−1029.849742 eV); `/var/tmp/jjlee-pah/d1/e0s/e0s_b3lypg_631gs_eV.json`.
  Whether the cation head uses these or fitted E0s is to be decided at training time.
- NeuralIL + EPNN: Mai et al.'s supplementary code under `src/Supplementary_Data_Code/lib/` (`bessel-nn-potentials-velo`,
  `epnn-main`) is the libraries only: no training scripts and no data format (`database_loading.py` is an example
  reader for another project's JSON). So no converter: a training script that reads the curated extxyz and calls
  `neuralil.training` *(not yet written)*; one model per charge state. Their EPNN predicts the dipole; for ions check
  that its charge constraint sums to the total charge.
- mace-jax: `mace-jax-train` with `--heads` (one head per charge state) and energy/force keys `dft_energy` /
  `dft_forces`; float32 training, TF32 off as in T0. Checked in container A (mace-jax 0.2.0): `--heads` takes a dict
  with per-head `train_path`/`valid_path`/`test_path`, `energy_key`/`forces_key` and E0s; `--E0s` takes a dict or a
  JSON file; a dipole loss exists (`energy_forces_dipole`). Dipole: EPNN as above, or a MACE dipole head built from
  `LinearDipoleReadoutBlock` and `DipoleSingleLoss` *(not yet written)*.
- Validation scripts: held-out errors per charge and `sample_T`; integrity runs with the MLMD driver
  (`d1/mlmd_run.py` with a charge-state model); harmonic frequencies from the model Hessian against PAHdb.
  *(The MLMD driver loads Mai et al.'s models; a model-loading hook for our models is needed.)*
- 2026-11-20 (project node): **AL round 1 ingested, dataset v3, v3 training started.** `al1_dft.tgz` received by
  tailcat (sha256 matches the Hangang entry); the 3 reference frames agree with the project node to 3.6e-12 Ha.
  `mlff_al_ingest.sh` (mlff_qa.py now has `--procs`; 64 processes on cores 80-143, output byte-identical to serial):
  2,423 frames, quarantine 200 = s2_review 197 + energy_outlier 2 (C15H12_328 f037, C19H14_349 f034: 1,200 K frames
  with a changed graph, 1.3 eV from xTB, both stable; xTB is the likelier one to be wrong there) + dipole_outlier 2;
  informational graph_changed 86, persistent_change 2, reoptimized 4. Kept as in v2 (only `unstable` dropped; none).
  The 76 dropped minima agree with the base labels to 1e-8 eV. `/var/tmp/jjlee-pah/d1/dataset_v3/`: train q1 2,612
  -> 4,959 (al1_md 300 K 325, 900 K 988, 1,200 K 1,034), train q0 2,610 and all held-out files byte-identical to v2.
  Training `mace_torch/v3` (v2 config, GPU 1, started 15:34, ~48 s/epoch, ~4 h).
- 2026-11-21 (project node): **v3 evaluated: cation forces better, neutral head slightly worse.** `d1/mlff_eval.py`
  on dataset_v3 + the AL batch (`/var/tmp/jjlee-pah/d1/eval/v3_vs_v2.{json,log}`). Force RMSE (meV/Å), v2 /
  v2_large_all / v3: validation q0 36.1 / 32.3 / 39.4, q1 92.0 / 94.6 / **82.5**; final_test q0 133.8 / 81.5 / 125.4,
  q1 250.6 / 236.7 / 227.8; transfer q0 41.2 / 43.9 / 40.9, q1 69.6 / 66.1 / 64.8; AL1 frames (now training data
  for v3) 82.4 / 80.3 / 44.4. Cation gain is across all config types (validation min 80.5 -> 69.2, md_300K 91.8 ->
  79.9) but the per-species gap remains: C12H8_385 349 -> 257, C12H10_202 213 -> 198, C11H10_320 117 -> 102.
  final_test q1 is dominated by C16H12_337 (1,022 -> 508) and C16H16_336 (900 -> 1,083). Neutral head regressed
  (validation q0 forces +3.3, every config type +3-5) with no change in its data: the shared layers moved toward
  the cations. q0 absolute energies regressed more (validation 10.0 -> 25.5, transfer 9.4 -> 37.6 meV/atom; E0s
  identical) but after the per-pair offset they are unchanged (0.8 / 0.4): a composition-dependent constant that
  forces and MD do not see. Conclusion: round 1 (more configurations of train species) gave ~10 % on cation
  validation forces; the remaining error is per-species (small, CH2-containing / non-planar cations). Round 2
  should add new species rather than more frames of the same ones (but see the next entry: PAHdb has few left).
- 2026-11-21 (project node): **AL round 2 prepared: 27 new small species, both charges.** Control first: v3 with
  seed 8 (`mace_torch/v3_seed8`, GPU 1) to see whether the q0 regression is beyond the seed spread. Pool
  (`d1/mlff_species_r2.py candidates`): PAHdb pairs with both charges and an unused skeleton are nearly exhausted
  (75 at <= 40 C, only 4 below 30 C). Taking instead every closed-shell RB3LYP/6-31G* neutral with <= 25 C whose
  carbon skeleton is in no split (v2 tables + transfer2): 27 pairs, 9–24 C, all train (4 five-ring incl. indene
  C9H8_495, 6 with sp3 C incl. a seven-ring C15H14_354 and triptycene-like C20H14_358, ~10 helicene-like non-planar).
  Not covered: biaryls (none in PAHdb at any size), and hydrogenated forms of held-out skeletons (hydrogenation keeps
  the carbon skeleton, so tetralin, indane-like and hydro-pyrenes stay in their parents' splits). PAHdb has a cation
  for only 4; the other 23 cation minima optimised here (`opt`: UB3LYP/6-31G*, GPU4PySCF b3lypg + geomeTRIC, SCF as in
  `dft_seed_run.py`, Stable=Opt at both ends; GPU 0, 3–10 min each): all converged, all stable at the end, ⟨S²⟩
  0.756–0.802, max |F| <= 0.013 eV/Å, bond graph equal to the neutral's. Frames: the seed recipe (`mlff_sample.py`
  defaults, `al2/sample_one.sh`), 32 per pair and charge: 1,728 (137 graph_changed), 331 stability-tested (g16_io
  write defaults). Reference frames labelled here; the optimised minimum C15H14_354_q1_f000 reproduces the
  optimiser's energy to 1e-9 Ha (max |F| 0.0005 eV/Å). Package `al2_hangang.tgz` (sha256 `17caf2a9…`), handoff
  `docs/plan/hangang_al2_handoff.md`. `mlff_al_ingest.sh`: keeps minima of new species (only repeats of base minima
  are dropped) and takes `SPECIES=` for the extra table.
- 2026-11-21 (project node): **non-Kekulé PAHs: q0 frames excluded, cations kept, triplet relabels.** PAHdb v4 lists
  334 non-Kekulé neutrals as RB3LYP; their ground state is a triplet (coordinator handoffs from glogin04 and z2mini:
  `d1/pahdb_v4_exclude_nonkekule_v1.csv` in gh200-containers, and a high-spin seed run of 10 new pairs, 528 frames).
  13 pairs in our data are affected (3 seed: C60H20_10225, C72H22_9911, C96H26_9397; the 10 high-spin pairs). MACE
  heads take no spin, so: (1) their q0 frames leave the q0 head (`d1/exclusions_nonkekule_q0.csv`, per-charge
  exclusions in `mlff_dataset.py`; v3 rebuilds byte-identically); (2) the 264 high-spin cation frames (doublets) join
  q1 after QA; `mlff_qa.py` now groups by (source, charge, mult) with ⟨S²⟩ limits per multiplicity (2: 0.80,
  3: 2.35; AL1 re-QA byte-identical). QA of the 528: no energy/dipole outliers, 2 re-optimised, 33 graph_changed;
  all 264 kept frames are s2_review (cations 0.80–1.08; the 14 triplets 2.19–2.47, held aside). (3) A q0t (triplet)
  head is a later experiment; the labels are being made now (`triplet_relabel/`, UKS mult 3, stability-tested,
  41 frames of the 3 seed pairs + f000).
  - C96H26_9397 f000 at the seed geometry: triplet −3674.542486 Ha (⟨S²⟩ 2.159, stable) is 7.44 mHa below the
    broken-symmetry singlet (−3674.535045, ⟨S²⟩ 1.058) and 24.76 mHa below RKS (−3674.517722): the q0 seed labels
    of these pairs are not the ground state, hence the exclusion.
  - Quartet check of the 10 high-spin cation minima (f000, UKS mult 4, `quartet_check/`): the doublet is lower in
    every case, by 9.1 (C54H20_10440) to 32.8 (C30H14_10770) kcal/mol, median 24; quartet ⟨S²⟩ 3.83–3.87, doublet
    0.80–0.83. The doublet q1 labels stand.
  - Triplet relabels done (42 frames: C60H20_10225 32, C72H22_9911 5, C96H26_9397 5; frames
    `triplet_relabel/frames_q0t`, headers `mult=3 ref_origin=triplet_relabel_v1 relabel_of=<pair>_q0_f<k>`): all
    converged without Newton, all stable, ⟨S²⟩ 2.15–2.38. The triplet is below the seed's closed-shell label at every
    frame: by 24.6–31.8 mHa (C60), 27.5–31.0 (C72), 15.8–27.5 (C96). ~0.1–0.3 GPU-h each on a shared GPU. Kept for
    the q0t experiment; not in v4.
- 2026-11-21 (project node): **seed control: v3's q0 "regression" is seed noise; the q1 gain is real.**
  `mace_torch/v3_seed8` vs v2, v2_seed8, v3 (`eval/seed8.{json,log}`, dataset v3 held-out sets + the AL1 frames).
  Force RMSE (meV/Å) v2 / v2_seed8 / v3 / v3_seed8: validation q0 36.1 / 41.2 / 39.4 / 34.7, q1 92.0 / 94.6 / 82.5 /
  83.0; final_test q0 133.8 / 147.3 / 125.4 / 126.3, q1 250.6 / 308.2 / 227.8 / 202.3; transfer q0 41.2 / 43.5 /
  40.9 / 41.8, q1 69.6 / 72.9 / 64.8 / 69.3; AL1 frames (q1) 82.4 / 83.2 / 44.4 / 37.4. The q0 energy RMSE swings
  with the seed (validation 10.0 / 19.2 / 25.5 / 7.9, transfer 9.5 / 30.2 / 37.5 / 12.0 meV/atom) but is all
  per-pair offset (0.7–0.8 and 0.3–0.4 after removing it): a constant per molecule that the seed sets, not a loss of
  physics. So the two-seed spread is ~5 meV/Å on q0 validation forces; q1 improves by ~10 meV/Å on validation and
  halves on the AL1 frames for both seeds. v4 should again be trained with two seeds.
- 2026-11-21 (project node): **AL2 labels from Hangang; bound_oop artefact; v4 build started.** Hangang labelled
  1,701/1,728 (handoff `glogin04-to-gh200-al2`, myhub `pah-data/al2_dft:2026-11-21`); the 27 frames its shards did
  not reach come from the local backup run. Overlap (1,577 frames): worst dE 2.3e-12 Ha, dF 1.3e-12 Ha/bohr, PASS.
  Hangang's two extreme-force frames (C20H14_358_q1_f010 80.7, C17H14_342_q1_f011 74.0 eV/Å) have C–H 0.65 / 0.63 Å:
  `mlff_sample.py` `bound_oop` pushes along the global plane normal, which in sp3 / non-planar molecules can drive an
  H into its carbon. Also in dataset v3: train q0 2, train q1 1, final_test q1 2 (C16H16_336, 0.62 / 0.69 Å). One of
  the final_test frames dominates final_test q1 force RMSE (v2 250.6 → 209.3, v3 227.8 → 168.9, v3_seed8 202.3 →
  178.3 meV/Å without the two), i.e. most of C16H16_336's apparent v3 regression. User decision: drop everywhere.
  `mlff_dataset.py` computes a `short_ch` flag (C–H < 0.75 Å) for `--drop-flags` (v3 rebuild without it
  byte-identical; with it exactly those 5 frames go; `dataset_v3_noshortch/` kept for comparisons). Sampler fixed: a
  push that shortens any distance (< 3 A) to the pushed atom below 0.8 x its value is redrawn; planar molecules never
  redraw (C20H12_395_q0 resampled byte-identical), the two bad pushes are replaced (shortest C–H now 0.96 Å).
  `mlff_al_ingest.sh` takes `EXCLUDE`, `REF`, `DROP`. v4 driver `al2/build_v4.sh` (unattended): merge, collect,
  ingest (BASE = v3's inputs + high-spin QA; SPECIES + species_r2, species_highspin; EXCLUDE non-Kekulé q0;
  DROP unstable,short_ch), train v4 with two seeds (GPU 0/1, waits if other users' GPU jobs appear), evaluate
  v3/v3_seed8/v4/v4_seed8 on dataset_v4, dataset_v3 and dataset_v3_noshortch.
- 2026-11-21 (project node): **G2 measured as defined (force MAE), v3: neutral passes, cation narrowly fails on a
  few small species.** `mlff_eval.py` now reports force MAE per set and per charge x config_type (held-out splits
  pooled), marked against G2 (60 meV/Å); per-frame `aF` in the JSON. v3 / v3_seed8 on `dataset_v3_noshortch`
  (`eval/v3_mae_noshortch.{json,log}`): MAE q0 validation 22.3 / 20.4, final_test 38.2 / 36.9, transfer 22.9 / 23.2;
  q1 43.3 / 41.1, 63.4 / 66.3, 39.1 / 39.4 meV/Å. Per config_type (pooled): q0 all <= 42.3 (md_1500K); q1 above 60
  only at md_1200K 61.0 / 62.3 and md_1500K 61.6 / 62.8 (300 K already 50). final_test q1 is per-species: C16H12_337
  302 / 476, C16H16_336 295 / 183, C10H12_332 227 / 279, C10H10_331 ~132, C14H12_340 ~117, C16H12_335 121 / 143 meV/Å
  (sp3 / five-ring small cations); without the top two 50 meV/Å. AL2 targets exactly this class; v4 tests it.
- 2026-11-21 (project node): **AL round 3, track 1 selected and packaged for Hangang** (user decision 2026-11-21
  23:35, via the coordinator: use Hangang's last allocation hours before the v4 evaluation is in). Pushed 7b11a9f
  first (sampler fix). `mlff_species_r2.py candidates --cmin 26 --cmax 40 --exclude-uids
  d1/pahdb_v4_exclude_nonkekule_v1.csv` (the list copied from gh200-containers 7de3421) against all species tables:
  848 pairs, of which 525 non-planar (oop RMS > 0.2 Å), all at 26–34 C. New `select`: the 2 motif pairs (C27H14_73
  sp3, C32H14_591 five+seven ring) first, then farthest-point sampling over nC, oop_rms, rings and H-edge shares;
  100 pairs ranked (26 C 8, 27 C 1, 28 C 16, 30 C 24, 32 C 31, 34 C 20; PAHdb cation for 1, 99 to optimise). Package
  `al3/al3_track1.tgz` (sha256 `a5ef1567…f27b`, myhub `pah-data/al3_track1:2026-11-22`, pull-back verified).
  Hangang optimises the cations, samples and labels (~1.2 GPU-h per pair for frames at 30–34 C, measured on AL1/AL2).
- 2026-11-22 (project node): **AL round 3, track 2: hand-built species outside PAHdb, packaged for Hangang** (user
  OK for an RDKit env, `/var/tmp/jjlee-pah/mamba/rdkit`, conda-forge). `mlff_species_own.py build`: 49 SMILES
  (five-ring, CH2/CH3/bridged/spiro sp3, biaryls) → RDKit + MMFF → GFN2-xTB; skeleton checked against all species
  tables. 37 accepted (33 new skeletons, 4 new H patterns on train skeletons, 9–25 C); rejected 3 held-out skeletons,
  7 molecules already in PAHdb, 1 duplicate, 1 SMILES that failed to parse. On gh200: B3LYP minima of both charges
  (`opt --charge 0|1`; 74/74 converged and stable in one round, 4–19 min each beside the v4 training, bond graphs
  unchanged), frames by the seed recipe with the fixed sampler (2,368 frames, shortest C–H 0.836 Å), manifest
  (416 stability-tested), 3 reference frames. Package `al3/own/al3_own.tgz` (sha256 `e618d2f8…072d`, myhub
  `pah-data/al3_own:2026-11-22`, pull-back verified). Hangang only labels (~15–30 GPU-h). At ingest, add
  `species_own.csv` to SPECIES. Replaced by `al3_own:2026-11-22b` (sha256 `e9b5c544…7b74`): frame headers carry
  `ref_origin=own_species_v1` (as the high-spin frames carry `own_highspin_v1`), manifest rebuilt, reference frames
  relabelled (identical energies).
- 2026-11-22 (project node): **v4 passes G2.** Force MAE (meV/Å), held-out sets of dataset_v4, v3 / v3_seed8 / v4 /
  v4_seed8: final_test q1 62.6 / 65.4 / 54.4 / 52.0, q0 38.2 / 36.9 / 32.0 / 29.7; validation q1 44.0 / 42.2 / 40.8 /
  40.1; transfer q1 39.4 / 39.7 / 39.0 / 38.2. Per charge × config_type (held-out pooled) every v4 entry is ≤ 60 for
  both seeds (worst q1 md_1500K 57.6 / 55.8, md_1200K 55.8 / 54.4; v3 61–63), a thin margin. Per species, the pairs
  above 60 are small hydrogenated / five-ring / biaryl held-out molecules (C16H12_337 271, C10H12_332 193, C10H10_331
  122, C16H12_335 121, C12H8_385 162, C12H10_202 99); large species incl. transfer q1 (62–100 C) are ≤ ~61. So the
  remaining error is where AL3 track 2 aims, not track 1 (non-planar 26–34 C): track 1's Hangang hours are the
  ones to cut. Next on v4: G4 (MD stability) and G5 (harmonic frequencies). Eval files `eval/v4_on_*`.
- 2026-11-22 (project node): **G5 frequency half on v4** (`mlff_accept.py symfreq`, as for v2, all 128 entries;
  `accept/symfreq_v4*.json`, `symfreq_v4_seed8*.json`). Strong-band RMS (cm⁻¹) median and pass (≤ 10), without the two
  pyrenes, v2 / v4 / v4_seed8: validation q0 5.8 (15/17) / 6.1 (15/17) / 5.8 (16/17), q1 5.4 (14/17) / 4.5 (14/17) /
  4.5 (16/17); final_test q0 6.2 (19/21) / 7.5 (15/21) / 5.4 (20/21), q1 7.1 (15/21) / 5.9 (15/21) / 6.2 (16/21);
  transfer q0 9.4 (13/24) / 10.7 (12/24) / 9.9 (12/24), q1 6.3 (17/24) / 6.8 (17/24) / 6.1 (17/24). All 128: median
  6.5 / 6.8 / 6.1, pass 93 / 89 / 97. **No change beyond seed noise**: the AL2 data did not move the harmonic
  frequencies; large neutrals stay the weakest group (C96H30_3317 16.5, C70H26_6628 16.8). Coronene⁺'s spurious
  −326 cm⁻¹ B3g mode is gone in both v4 seeds (only flat −3 … −11 cm⁻¹ modes of large molecules remain).
  Outlier C16H12_337 (dihydropyrene, held out): q0 29 → 64 / 92, driven by the aliphatic CH2 stretch at 2,883 cm⁻¹
  (+186 / +290) — the sp3 CH2 environment that also gives it the worst force MAE (271); AL3 track 2 adds CH2 species.
  C10H12_332⁺ 34.6 → 29.5 / 31.7. Intensities (the other half of G5) still need a dipole model.
- 2026-11-22 (project node): **AL3 track 2 labelled on Hangang, ingested: dataset_v5.** `al3_own_dft:2026-11-22`
  (2,368/2,368 frames, array 1335773, 29.3 GPU-h; Hangang's labels of the 3 `ref/` frames match gh200's to 1.8e-12 Ha)
  received 12:05 (handoff `2026-11-22-pah_stage0-glogin04-to-gh200-al3own`). Ingest driver `al3/ingest_own.sh`: v4's
  inputs + v4's AL2 QA file as BASE, SPECIES + `al3/own/species_own_ok.csv`, same EXCLUDE / DROP (unstable,short_ch),
  REF dataset_v4; `mlff_al_ingest.sh` now takes `CORES`/`PROCS` (QA on 64-95,128-143, off the local G4 runs). QA: 12
  quarantined (6 `s2_review`, 6 `dipole_outlier`, all cations), 174 graph_changed, 296 boundary, 5 persistent_change,
  1 re-optimised; 74 new minima, nothing dropped by flags. Train q0 3,431 → 4,615, q1 6,012 → 7,196 (+1,184 each);
  every held-out file byte-identical to v4's. **One bad frame kept by the current rules**: C10H10_x02_q1_f007
  (2-methylindene⁺, nms_1000K): one H 2.19 Å from the nearest C (the other 31 frames ≤ 1.4 Å), E 12.75 eV above the
  minimum (next highest 4.1 eV), ⟨S²⟩ 1.265 — a dissociated H with a spin-contaminated UKS label, in train_q1 because
  `s2_review` is flagged, not dropped. **Dropped by user decision** (same day): `mlff_dataset.py --drop-frames`
  (CSV of single frames with reasons, flag `dropped_frame`, stops if a listed frame is missing), list
  `d1/drop_frames_v5.csv`, `DROP_FRAMES` in `mlff_al_ingest.sh`; dataset_v5 rebuilt: train q1 7,195, every other file
  byte-identical to the first build. The milder `s2_review` frames stay (rule unchanged).
- 2026-11-22 (project node): **G4 on v4, local seed-0 subset (900 / 1200 K, 16 held-out pairs × 2 charges, 32 runs,
  20 ps NVT + 50 ps NVE).** Scored with `d1/g4_score.py` (new: bond graph of every saved frame, 10 fs apart, against the
  relaxed start; a bond change counts only if it holds in every frame of the last 100 fs, per bond, since C-H bonds
  flicker over the 1.2 × covalent cutoff at 1200 K; also blow-up time T > 3 × target and NVE drift); output
  `accept/g4_v4_local_score.json`. Intact: 900 K 15/16 (93.8 %, G4 ≥ 99 % FAIL), 1200 K 14/16 (87.5 %, ≥ 95 % FAIL).
  **All three failures are C16H12_337** (trans-15,16-dihydropyrene, final_test): its two interior methine H atoms
  (C0-H1, C6-H7) leave and the core becomes pyrene (C0-C6 1.57 → 1.43 Å). q1 900 K: T blows up at 15 ps (NVT), the
  whole molecule flies apart; q1 1200 K: both H ejected at ~10.8 ps with a T spike > 3,600 K, flying off to hundreds of
  Å (an unphysical energy release, not thermal loss); q0 1200 K: both H leave within 5-20 ps, one re-binds at rim C13,
  NVE drift 5.9 meV/atom (all other runs |drift| median 0.14, max 0.6). The other 15 species, both charges, stay
  intact at both temperatures. The same species has v4's worst held-out force MAE (271 meV/Å) and the G5 CH stretch
  outlier: the model fails on sp3 C-H inside a PAH core. dataset_v5 adds hand-built sp3 species but no interior-H
  ones; whether v5 fixes it is an MD question (rerun this species after v5). Hangang's 192 runs will give the 4-seed
  picture and the 50/300 K rows.
- 2026-11-22 (project node): **Dipole feasibility test: AtomicDipolesMACE per charge on dataset_v4 fails the G2 dipole
  limits.** `d1/mace_torch_dipole.yaml` (v2-sized: 64x0e+64x1o, 2 interactions, lr 0.01, 300 epochs, EMA), label
  μ − q·R_COM; trained 19:13–21:16 beside two root GPU jobs (user request); eval `eval/dip_v4.{json,log}` (`d1/mlff_dipole_eval.py`).
  Component MAE (D), model / zero-model: q0 validation 0.159 / 0.331, final_test 0.161 / 0.298, transfer 0.158 / 0.313,
  train (first 600) 0.098 / 0.321 (G2 ≤ 0.06: only `min` frames pass, 0.036); q1 validation 0.517 / 0.673, final_test
  0.444 / 0.576, transfer 0.584 / 0.719, train 0.302 / 0.542 (G2 ≤ 0.1: every config_type fails, 0.27–0.63).
  Translation invariance holds (3 Å shift ≤ 1e-5 D). **q1 error grows with size** (held-out MAE 0.15 D under 20
  atoms, 0.34 at 20–39, 0.44–0.80 above 40, i.e. 65–80 % of the zero model for large cations; worst C60H30_3608 1.61,
  C72H22_9911 1.37): the expected failure of a sum of bounded local atomic dipoles for a +1 charge spread over the
  molecule, whose Σ qᵢ(rᵢ − R_COM) term grows with size. q0 halves the zero-model error at every size but stays 2.5×
  over the limit. Both models underfit (train error close to held-out; q1's best validation at epoch 53 of 300, then
  drifting up), so the size of the gap is not final, but the q1 size trend is structural. Next options: (a) a
  charge model with the total-charge constraint (atomic charges summing to q, plus atomic dipoles: μ = Σ qᵢ(rᵢ − R_COM)
  + Σ pᵢ), e.g. EPNN as planned or a MACE charge head; (b) retune the local model (lr 1e-3, larger) only as a q0 check.
- 2026-11-22 (project node): **G4 on v4, Hangang + local, 160 of 192 runs: v4 fails G4, on two species only.**
  Hangang (KISTI PyTorch 25.03 + mace-torch 0.3.17, `pah-data/g4_v4_out:2026-11-22`; 4 array tasks, queue waits
  3 h 25 m – 3 h 54 m, each hit the 6 h limit) finished 122 runs, 6 were cut off, 32 (seed 3 at 300/900 K) never started
  (Slurm deadline); gh200 ran 32 (seed 0, 900/1200 K). Scored with `d1/g4_score.py` (per-bond persistence over the last
  100 fs; this clears 6 runs whose only change was one C-H past the cutoff in the last frame); the 6 cut-off runs all
  show persistent breakup in their partial .traj and count as failures. Intact: 300 K 48/48, **900 K 44/48 (91.7 %,
  ≥ 99 % FAIL), 1200 K 54/64 (84.4 %, ≥ 95 % FAIL)**. Without C16H12_337 (trans-15,16-dihydropyrene): 42/42, 42/42,
  54/56 (96.4 %, passes); without it and C10H10_331: all intact. C16H12_337 fails 12 of its 14 runs at 900/1200 K:
  interior H ejection and blow-ups (q1), and in q0 a lone C0-C6 cleavage (the central C15-C16 bond; 900 K s2, 1200 K
  s2/s3, NVE drift ≤ 1.3 meV/atom), which may be the real thermal ring opening to the cyclophanediene and needs a DFT
  barrier check before it is called a model failure. C10H10_331⁺ blows up at 1200 K in 2 of 4 seeds (all bonds lost at
  12 ps in s1). NVE drift median 0.18 meV/atom (max 100.9 in a C16H12_337⁺ run). Score files
  `accept/g4_v4_{hangang,local}_score.json`.
- 2026-11-22 (project node): **Hangang work taken over by gh200 (user); dataset_v6; targeted species for the G4
  failures.** Not continued: G4 on v4 seed 3 / the 6 cut-off reruns (cannot change v4's verdict) and AL3 track 1 ranks
  33–100 (~75 GPU-h; v4 says large non-planar species are already fine). (1) **dataset_v6** = v5 inputs + track 1
  ranks 1–32 (`al3/ingest_track1.sh`; species_r3, REF dataset_v5): QA 83 quarantined (48 `s2_review` cations ≤ 0.875,
  32 `unstable`, 3 `dipole_outlier`). **The 32 `unstable_rhf_uhf` neutrals (7 species, all q0) are kept with their RKS
  labels by user decision**, an exception to the 2026-11-17 drop-unstable rule (which still drops the 11 earlier
  `unstable` frames): `mlff_dataset.py --keep-frames` (CSV frame, flag, reason; exempts that frame from that drop flag
  only), list `d1/keep_frames_v6.csv`, `KEEP_FRAMES` in `mlff_al_ingest.sh`. Train q0 4,615 → 5,639, q1 7,195 → 8,219;
  held-out files byte-identical to v5's. (2) **Targeted species** (`d1/species_g4_targets_smiles.csv`,
  built with `mlff_species_own.py build --prefix y` into `/var/tmp/jjlee-pah/d1/al4t`): both failing species are
  hydrogenated held-out skeletons (pyrene, naphthalene; final_test), so only new or train skeletons qualify. 13 of 19
  accepted: interior ring-fusion H — 2-methyl-, 2,7-dimethyl-, 4-methyl-, 2-tert-butyl-trans-15,16-dihydropyrene,
  2-methyl-4a,8a-dihydronaphthalene, 2-methyl-4a,10a-dihydrophenanthrene; bis-allylic/benzylic CH2 (the C10H10_331
  1,4-dihydronaphthalene motif) — 2-methyl-, 6-methyl-, 2,3-dimethyl-1,4-dihydronaphthalene, 2-methyl- and
  9-methyl-9,10-dihydroanthracene, 2-methyl-1,4-dihydroanthracene, 3-methyl-1,4-dihydrophenanthrene. Rejected:
  held-out anthracene/phenanthrene skeletons (5), 4H-cyclopenta[def]phenanthrene (in PAHdb). Driver `al4t/run_al4t.sh`
  (track 2 recipe: B3LYP minima of both charges, graph check, 32 frames per pair and charge, manifest, labels here in
  8 shards over both GPUs, collect), started 22:02.
- 2026-11-23 (project node): **dataset_v7, v7 (two seeds): G2 passes with more margin; G4 first pass shows no net gain.**
  Targeted species labelled 832/832 (22.1 GPU-h; 1 neutral `unstable_rhf_uhf` dropped by rule, 1 cation ⟨S²⟩ 0.808
  kept). dataset_v7 = v6 + these: train q0 6,054, q1 8,635; held-out files byte-identical to v4's. `mace_torch_v2.yaml`,
  seeds 7/8, 300 epochs. Force MAE (meV/Å) v4 / v4_seed8 / v7 / v7_seed8: final_test q0 32.0/29.7/20.5/22.4, q1
  54.4/52.0/42.3/43.1; transfer_test q0 20.8/20.1/21.0/22.5, q1 39.0/38.2/40.0/39.6; worst config_type (q1 md_1500K)
  57.6/55.8/51.0/51.9. The final_test gain sits on C16H12_337, C10H12_332, C10H10_331, the species whose G4 failures
  chose the targeted species: final_test is no longer blind for these three; transfer_test (unchanged) is the blind
  figure. G4 first pass (the three species × q0/q1 × 900/1200 K × seeds 0–1, 23 of 24 scored): 900 K 12/12, 1200 K
  7/11. v4 on the same jobs: 4 failures; v7: 4, different ones. C16H12_337 q1 now intact; q0 1200 K s0 still moves
  interior H. New: the CH2–CH2 C–C bond (C4–C8) breaks in C10H10_331 (q0 s1, q1 s0, 1200 K) and C10H12_332 q1 1200 K
  s0 loses C2–C6 and C4–C8 (C2H4 leaves).
- 2026-11-23 (project node): **DFT along the four v7 failure paths: the breaks are model artefacts, in v4 as much as
  v7.** `d1/g4_fail_frames.py` picks 22 frames per failed run (MD start, 3 early thermal frames, 18 frames with the
  stretch s = d/cutoff of the changing bonds spread from 0.8 to 2.5 over the first break held 100 fs);
  B3LYP/6-31G* labels (88 frames, 6.8 GPU-h) and a broken-symmetry UKS check of the 44 neutral frames (`run --stab-all
  --bs`); `d1/g4_path_energy.py` compares energies relative to the MD start. Near the start (s < 1) every model is
  within 0.05 eV of DFT. With the bond broken (s > 1.1), model minus DFT (BS-UKS where lower), mean eV, v4 / v7 /
  v7_seed8: C10H10_331 q1 −0.70/−0.60/−0.52; C10H10_331 q0 −1.04/−1.19/−1.14; C10H12_332 q1 −0.70/−0.76/−0.54;
  C16H12_337 q0 −1.37/−0.98/−1.16; force MAE there 170–700 meV/Å. All models put the broken region 0.5–1.4 eV too
  low, so v4's intact runs on these jobs were luck, not a better surface. Neutral broken frames: RKS unstable (19),
  BS-UKS up to 1.2 eV lower, ⟨S²⟩ up to 1.1 (biradical); under the standing drop rule such frames need the BS label
  (`collect --bs-label`) to enter training. These frames are of held-out (final_test) species, so they are a diagnosis,
  not training data. Output `/var/tmp/jjlee-pah/d1/al5g4/` (`dft.extxyz`, `dft_bs.extxyz`, `path_energy.{json,log}`).
- 2026-11-23 (project node): **Course corrections (user, via the coordinator note z2mini 2026-11-23 21:06).**
  (1) Reactive sp3 bond-breaking AL stops after v8 (the running `al5/chain_v8.sh`); no v9 reactive round. If v8
  passes the 48-run 1200 K G4 check, v8 is frozen as the D1 force model. If the labile sp3/dihydro species
  (C16H12_337, C10H10_331, C10H12_332) still rearrange at 1200 K while sp2 aromatic PAHs stay 100 % intact and blind
  transfer_test stays ~39–40 meV/Å, D1 production scope is stratified: sp2 aromatic PAHs up to 900–1200 K,
  labile sp3/hydroaromatic PAHs capped at ≤ 600–900 K. (2) q0 surface consistency: dataset_v6 keeps 32
  `unstable_rhf_uhf` neutral frames with RKS labels (`d1/keep_frames_v6.csv`) while v8 adds stretched neutral
  frames with BS-UKS labels to the same spin-blind q0 head; compare v8 with v7 on q0 held-out forces and neutral G4,
  and look for Coulson–Fischer kinks; if q0 degrades, drop `keep_frames_v6.csv` (or keep v7 for sp2 production).
  (3) G2 energy criterion on offset-removed energy RMSE (Phases, above). (4) Before D1 production MLMD, the MACE MD
  runner batches 16–32 independent NVE replicas of 25–50 ps per GPU (seeded from decorrelated Langevin snapshots,
  frames every 25–50 fs, cuEquivariance on) instead of serial 4 × 200 ps single-molecule runs (~95 steps/s/GPU).
  (5) Datasets and final models archived to myhub `pah-data`: `dataset_v7:2026-11-23` (dataset_v7 + _work; tgz sha256
  `f9f61515…bc65f`, manifest digest `sha256:99382642…aefab`), `mace_v7:2026-11-23` (v7, v7_seed8: model, compiled model,
  config, logs, results, training logs, eval/v7_on_v7; tgz sha256 `d5a73b90…48336`, digest `sha256:7b2680fe…0d48b`);
  pull-back verified. dataset_v8 and v8 follow when the chain ends.
- 2026-11-24 (project node): **v8 round, steps 1–4.** v7 hot MD (1500 K, 5 + 20 ps, 41 trained ring-sp3 pairs ×
  2 charges): intact q0 34/41, q1 19/41 (an AL probe, not a gate); 29 runs with a persistent change → 406 frames along
  the break paths (`g4_fail_frames.py --al-tag al5`, 14 per path); 73 stretched/broken neutrals labelled with
  broken-symmetry UKS (32 took the BS energy, 41 stayed RKS), 333 with the normal labelling (31 GPU-h; 40 doublets with
  ⟨S²⟩ > 0.80 quarantined by QA as usual). **Label-consistency stop:** the ingest's repeated-minimum check showed
  C13H10_55 q0 frame 0 0.67 eV above its dataset_v7 minimum at the same geometry: that species' minimum is
  RKS-unstable and dataset_v7 labels its near-minimum frames with BS-UKS (⟨S²⟩ ≈ 1.0), but the al5 split sent only
  q0 frames that are broken or stretched (s ≥ 0.95) to the BS run, so its three early thermal frames got RKS labels.
  Also C27H14_73 q1 frame 0 converged 0.10 eV above the v7 minimum at the same geometry (another UKS solution); its one
  untested near-start frame is suspect. v8 training (started 03:02) was stopped after 4 min, those 4 frames dropped
  (`d1/drop_frames_v8.csv`), dataset_v8 rebuilt (train q0 6,142, q1 8,920; held-out files = v7's) and training
  restarted (`al5/chain_v8b.sh`, steps 5–7 of chain_v8.sh). `mlff_al_ingest.sh` now prints WARNING when a repeated
  minimum differs from the base one by > 0.05 eV (the other two cases, C15H14_y19 and C11H12_y05 q1, 0.08 eV, are
  other conformers: 1.3 and 0.6 Å). Rule for any later q0 labelling: for a species whose minimum is `bs_relabel`, every
  q0 frame goes to the BS run. C13H10_55 q0 already spans two regimes in v7 (its 600 K MD frames, RKS-stable, lie
  below the BS minimum: another isomer), so it joins C10H10_331 on the Coulson–Fischer kink-scan list (item 2).
- 2026-11-24/25 (project node): **v8 (two seeds) and its G4.** G2 (force MAE meV/Å, v7 / v8 / v8_seed8): final_test
  q0 20.5 / 21.0 / 21.5, q1 42.3 / 43.3 / 43.4; transfer_test q0 21.0 / 21.1 / 21.0, q1 40.0 / 38.5 / 39.6; worst q1
  config_type 51.0 / 50.8 / 51.1; offset-removed energy RMSE 0.3–1.1 meV/atom: pass. q0 per config_type within
  ±0.6 meV/Å of v7, raw q0 energy RMSE lower (final_test 12.2 → 7.6): the BS-UKS stretched-neutral labels did not
  degrade the neutral head on held-out data. Failure paths (al5g4, not trained on), model − DFT at broken frames:
  cations from −0.6…−1.1 eV (v7) to ±0.3 eV; neutrals from −1.0…−1.5 eV to −0.4…−0.9 eV (vs BS-UKS where lower).
  **G4 at 1200 K on v8** (C16H12_337, C10H10_331, C10H12_332 × q0/q1 × seeds 0–7, 20 ps NVT + 50 ps NVE,
  `accept/g4_v8_score.json`): q0 24/24, q1 23/24, all 47/48 = 97.9 % (≥ 95 %): **pass**. The one failure,
  C16H12_337 q1 seed 1, is two H shifts (C0→C13, C6→C12) at 68.5 ps, no C–C break; |NVE drift| median 0.17, max
  0.94 meV/atom over 50 ps. v4 and v7 failed 4–5 of the 24 1200 K runs of the first pass. Per the course correction,
  v8 is the D1 force model candidate (freeze after the regression G4 on the other species and the q0 kink scan);
  no v9 reactive round. Archived to myhub `pah-data`, pull-back verified: `dataset_v8:2026-11-25` (dataset_v8 +
  _work; tgz sha256 `ad1e3d04…292d27`, digest `sha256:600ca085…dc10a0`), `mace_v8:2026-11-25` (v8, v8_seed8: model,
  compiled model, config, logs, results, training logs, eval/v8_on_v8, g4_v8_score, al5g4 path_energy_v8; tgz sha256
  `44e49a89…ea746bb`, digest `sha256:a768f7a5…37c554`). Batched MLMD (`d1/mace_md_batch.py`, C42H18_105 q1, 600 K):
  cueq + 512 replicas 13,254 replica-steps/s (serial ASE ~95); without cueq, batches > 30,720 atoms are refused
  (1024 × 60 atoms corrupted replicas ≥ 318).

## Phase 3 working notes

- Selection per round: ensemble MD (4 models) for ~500 species-charge pairs × 300–1500 K, 20 ps each; frames every
  0.1 ps; keep the top ~2 % by maximum force disagreement per molecule, de-duplicated (at most ~10 per trajectory).
- Each round ~10k Gaussian points; retrain from scratch or fine-tune (to decide by comparing on round 1).
- Stop criteria: on never-trained species, integrity failures < 1 % at ≤ 900 K and < 5 % at 1200 K, and the
  95th percentile of ensemble force disagreement below the G2 force threshold.

## Tools still to write

| Tool | Phase |
| --- | --- |
| NeuralIL/EPNN training script over the curated extxyz (`d1/mlff_dataset.py` output) | 2 |
| Model-loading hook in `d1/mlmd_run.py` for our trained models | 2 |
| Harmonic frequencies and intensities from a model, against PAHdb (G5) | 2 |
| MACE dipole model or EPNN for ions | 2 |
| ~~Active-learning selector~~ written: `d1/mlff_al_select.py` (2026-11-20) | 3 |

