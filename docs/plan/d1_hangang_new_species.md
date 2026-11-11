# D1 MLMD on Hangang: overlap with Mai et al. and new species (plan, 2026-11-11)

Decided 2026-11-11, after the Hangang calibration (job 1257396; `cluster_migration.md`, "Step 4 calibration on
Hangang"). Re-running Mai et al.'s published grid (1,704 species × 50/300/600 K × 4 replicas, ~470 GPU-h) adds
little, so it is **not** run. Hangang runs instead:

1. a small **overlap** with Mai's grid, to tie our spectra to theirs and to measure the scatter between seeds;
2. **~1,000 new species** that fill gaps in D1's coverage, at **300 and 600 K** only for now.

900 and 1200 K wait for the model-stability test running on the project node; the new species and the overlap set
get those temperatures later if the model holds there.

## Coverage of Mai's 1,704 (profiled from their xyz files, 2026-11-11)

Bond graph from covalent radii; adjacency from runs of bonded aromatic CH carbons; pentagons are chordless 5-cycles.

| D1 target (`big_picture_proposal.md`, D1) | Mai's 1,704 (PAHdb v3.20 neutrals) |
| --- | --- |
| Nc 20–200 evenly | 876 (51 %) at Nc 30–39, 725 of them four isomer families (C33H19, C32H18, C34H20, C31H17). Nc 10–19: 76; 20–29: 304; 40–49: 50; 50–59: 61; 60–79: 142; 80–99: 79; 100–129: 42; 130–159: 66 (34 are C144H36); 160–216: 4. 223 distinct formulas |
| Solo/duo/trio/quartet from compact to irregular | Dominant class among the 1,664 pure C/H: quartet 781, trio 417, duo 342, solo 124 |
| Hydrogenation, fully dehydrogenated to +4 H | 79 with sp3 C (superhydrogenated), 19 with bare edge C |
| Pentagons | 23 |
| Heteroatoms (held out as an OOD test set, M1) | 40 with N, none with O |

## Model domain

- Elements C, H, N, O only (`sorted_elements` in their code). PAHdb species with Mg, Si or Fe are excluded.
- Trained on 687 neutral PAHdb v3.2 species, 300 K classical MD plus Gaussian perturbations, B3LYP/4-31G; 350 of
  17,175 configurations superhydrogenated (Mai+2025 §2.1). Neutral, closed- or open-shell as in PAHdb.
- PAHdb v4.00 recomputed everything at B3LYP/6-31G* (Ricca+2025 §2.1). Its geometries are only starting
  points (the 200 ps NVT re-equilibrates them on the model's surface). UIDs are not assumed to match v3.20, so
  overlap with Mai's set is found by structure (formula and bond graph), not by UID.
- v4.00 added 4,651 pure-PAH spectra (edge defects such as bay regions; most at Nc ~50, then ~70 and ~100) and
  1,866 N-PAH spectra (most at Nc ~100, up to three N), all charge states (Ricca+2025 §2.2). Many lie outside the
  model's training set, hence the domain screen (step 2).

## Steps

**0. PAHdb v4.00 on Hangang.** `pahdb-complete-theoretical-v4.00.xml.gz` (sha256 in `t0/pahdb-v4.00.sha256`;
download needs PAHdb's e-mail form) copied to `/appsdata/$USER/d1/pahdb/` through a dm node, hash checked.

**1. Selection** (`d1/select_new_species.py`, numpy only, in container A).
- Pool: neutral PAHdb v4.00 species with only C, H, N; not in Mai's 1,704 (structure match).
- Cells: Nc bin × dominant adjacency (solo/duo/trio/quartet) × class (pure benzenoid, pentagon-containing,
  superhydrogenated or with CH2/CH3, dehydrogenated, N-PAH).
- Fill: 1,000 species, the emptiest cells first (counting Mai's species already in them), weighted to Nc 40–200
  and to solo/duo-rich compact structures; at most a few species per formula so no isomer family dominates;
  **N-PAHs 10 % (100 species)**, spread over Nc.
- Output: `d1/new_species_v1.csv` (UID, formula, class, cell, adjacency counts) and the xyz files, in Mai's format,
  in `/appsdata/$USER/d1/XYZ_new_v1/` (PAHdb-derived data, kept out of git). The cell table is reviewed before
  step 2.

**2. Domain screen** (driver change, so a new job: one small test job first).
- NNFF forces at the v4 geometries, compared with the same statistic for Mai's species at theirs.
- A short 600 K run (about 10 ps NVT + 5 ps NVE) with the driver's flags (non-finite, T > 30,000 K, > 25
  neighbours, edge overflow) plus a new bond-topology check at the end (no C–H or C–C bond broken or formed).
- Criteria to match the high-T stability test on the project node. Species that fail are dropped and listed,
  not tuned; the selection refills their cells from the next candidates.

**3. Overlap run.** 48 of Mai's species, stratified by Nc and class (including a few N-PAH, pentagon and
superhydrogenated ones), × 300 and 600 K × **8 replicas**; 200 ps NVT + 200 ps NVE.
- Each published 300 and 600 K spectrum ranked among our replicas (strong-band centroids, band-window shares, as
  `mai_spectrum_compare.py` does for naphthalene).
- Seed scatter of band shares as a function of Nc; it sets the replica count of step 4 (4 by default; fewer above
  an Nc where the scatter is small).
- ~21 GPU-h (~5–6 node-hours).

**4. New-species production.** ~1,000 species × 300 and 600 K × 4 replicas (or as step 3 sets),
200 ps NVT + 200 ps NVE.
- ~290 GPU-h (~72 node-hours) at the Hangang calibration speed (1.22 s per atom per system, plus ~250 s per unit).
- A 5-task array with 6–12 h tasks, resubmitted to resume: on 2026-11-11 half of `normal` was held for large
  jobs, and short tasks backfill much sooner than 24 h ones.
- Dipoles ~2.4 MB per system, ~20 GB in all, in `/appsdata`.

## Licence

The NNFF and EPNN models (and Mai's data) are CC BY-NC-SA 4.0, so the MLMD spectra made with them are released
under CC BY-NC-SA 4.0, as an adaptation, with attribution to Mai et al. 2025. The rest of D1 can keep the
proposal's open licence; the MLMD part is a separately licensed component. Run outputs stay out of git.
