# Charged-state MLFF: Gaussian training data, training, active learning (draft, 2026-11-12)

Status: **tools built and Phase 0 inputs ready; nothing submitted on Hangang.** Branch `charged-mlff`.
Reviewer-facing version (self-contained, with evidence, gates, budget and questions):
`docs/plan/charged_mlff_proposal.md`. This file is the run-book.
Context: `docs/plan/charged_tdep_validation.md` (the xTB test: band positions transfer from neutrals to cations,
intensities unsettled; the DFT timing pilot) and `docs/plan/cluster_migration.md` (neutral MLMD, Hangang).

## Goal

Force fields (and dipole models) for PAH neutrals and cations, later anions and dications, that stay stable and
accurate from 50 to ~1200 K, to compute temperature-dependent spectra per charge state. Mai et al.'s model is
neutral-only, trained on 300 K geometries, and breaks above ~600 K (and for some species at any temperature).

## Facts this plan rests on

- Hangang has Gaussian `gaussian/g16.c02` (module, commercial tree). `normal_cpu`: 680 nodes, 174,080 CPUs (256 per
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

**Phase 0: Gaussian on Hangang (inputs ready; < 100 core-h).** `d1/g16_phase0.py` →
`/var/tmp/$USER-pah/d1/g16_phase0/` (28 inputs, `expected.json`):
- `timing/`: the PySCF pilot's 12 geometries (C10–C144, neutral and cation, same displacement) at 4-31G and 6-31G\*.
  4-31G energies must match PySCF `b3lypg` (expect ≤ ~1e-5 Ha: grids differ); gives Gaussian's real cost per size.
- `origin/`: the naphthalene cation and the same geometry shifted by +3 Å in x. If Gaussian's dipole is about the
  coordinate origin, the two differ by 3 e·Å = 14.41 D in x; then `g16_io.py parse --origin coords` (default) is
  right; if they agree, use `--origin nuclear_charge`.
- `pahdb/`: naphthalene neutral (uid 330) and cation (uid 359) at their PAHdb geometries, 6-31G\* Freq: energies
  against PAHdb `total_e` (−385.89270636 / −385.61361154 Ha) and frequencies against PAHdb's stored values.

On Hangang (copy the directory to `/appsdata/$USER/d1/g16_phase0`, then):
```bash
cd /appsdata/$USER/d1/g16_phase0
for n in 8 32; do grep -l "nprocshared=$n" $PWD/*/*.gjf > list$n.txt; done
sbatch --array=0-19 --cpus-per-task=8  --mem=20G --time=04:00:00 /appsdata/$USER/pah_stage0/d1/slurm_g16.sbatch list8.txt 1
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
- DFT: `d1/g16_io.py write` (one `.gjf` per frame, `manifest.json` with frame info) → `d1/slurm_g16.sbatch` →
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

## Quality checks after each DFT batch (gate G1 in the proposal)

1. Termination: count logs without "Normal termination" (`parse_failures.txt`); resubmit once; then drop.
2. Spin: histogram of `s2` for doublets; flag > 0.80, list by size.
3. Outliers: per species and charge, energies against the median of its frames (a failed SCF often lands on a
   wrong state, several eV away); forces with any component > 10 eV/Å.
4. Force consistency on ~10 frames: central finite differences of the energy along a few coordinates against the
   analytic force (≤ 1×10⁻⁴ Ha/bohr). *(Tool not yet written.)*
5. Coverage: points per charge, size bin, `config_type`; held-out species must have no training frames.

## Phase 2 working notes

- Splits: by species (`split` column of `species.csv`); never split frames of one species between train and test.
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
| Force finite-difference check (G1) | 1 |
| Data QA report (termination, ⟨S²⟩, outliers, coverage) | 1 |
| Extended XYZ → NeuralIL/EPNN training format | 2 |
| Model-loading hook in `d1/mlmd_run.py` for our trained models | 2 |
| Harmonic frequencies and intensities from a model, against PAHdb (G5) | 2 |
| MACE dipole model or EPNN for ions | 2 |
| Active-learning selector (ensemble MD, disagreement, de-duplication) | 3 |

