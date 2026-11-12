# Charged-state MLFF: Gaussian training data, training, active learning (draft, 2026-11-12)

Status: **tools built and Phase 0 inputs ready; nothing submitted on Hangang.** Branch `charged-mlff`.
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

## Decision needed: level of theory

| | B3LYP/6-31G\* | B3LYP/4-31G |
| --- | --- | --- |
| Consistent with | PAHdb v4 (harmonic library, geometries, all charge states) | Mai et al.'s model and published spectra |
| Seed-set cost (5,800 points, PySCF-based estimate) | ~10k core-h | ~3k core-h |
| Full data (50–80k points, ≤ 100 C) | ~85–220k core-h | ~25–65k core-h |

Mai's data are not available, so the neutrals are retrained either way. Recommendation: **6-31G\***, so the
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

**Phase 1: seed set, neutrals and cations (~10k core-h at 6-31G\*; about a day on 5 nodes).**
- Species: `d1/mlff_species.py` → `/var/tmp/$USER-pah/d1/mlff_seed/species.csv`: 116 closed-shell neutral /
  doublet cation pairs of the same structure (bond-graph hash), 10–60 C, ~23 per 10-C bin, farthest-point
  sampling in solo/duo/trio/quartet fractions; naphthalene, anthracene, phenanthrene, pyrene, coronene, ovalene and
  circumcoronene always included; 23 pairs held out. Anion and dication uids recorded for later. Small pairs are
  scarce in PAHdb (25 with 10–19 C, 27 with 20–29 C), so nearly all of them are in.
- Geometries: `d1/mlff_sample.py` (GFN2-xTB Langevin MD, FixCom): per species and charge 5 displaced minima
  (0.05 Å) and 4 frames each at 300/600/900/1200/1500 K (2 ps equilibration, 0.5 ps apart); frames whose bond graph
  changed are dropped. 25 frames × 232 = 5,800 points.
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
