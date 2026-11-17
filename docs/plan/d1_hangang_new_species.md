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
- **Done 2026-11-11.** PAHdb v4.00 (downloaded on Hangang, sha256 matches `t0/pahdb-v4.00.sha256`) has 4,479
  neutrals; 46 hold other elements, 1,698 are Mai's species (1,696 of Mai's 1,704 found by structure, 1,690 with
  the same UID; Mai's set itself has 8 duplicate structures, e.g. C18H12_211/2356/4015), 24 are duplicate
  structures in v4, 4 are too small or large.
  - Structure key: plain 1-WL graph hashing gave 59 false collisions among Mai's benzenoid isomers; seeding each
    atom's label with its graph-distance profile leaves only true duplicates (identical geometry).
  - **Domain filter (803 dropped):** Mai's species all have H, at most 2 bare edge C, 2 pentagons and 12 sp3 C.
    v4's extra neutrals include pure-carbon clusters and cages (C26–C34 with no H, many triplet or quintet, two
    with 12 pentagons) that the greedy fill otherwise picked as "dehydrogenated" and "pentagon" species. The
    candidates must lie inside Mai's range (`in_domain`).
  - In-range pool: 1,904 (479 N-PAH). v4's additions are isomer sets (144 formulas), so 1,000 species need up to
    40 per formula (20 gives 828; the extra ones land at Nc 40–79, the main gap).
  - Chosen: 1,000 (100 N-PAH), 83,250 atoms, 94 atom counts, at most 256 atoms; `d1/new_species_v1.csv`,
    `d1/cells_v1.csv`. Dominant adjacency: duo 579, solo 259, trio 104, quartet 58 (Mai's set is quartet-heavy).
    Multiplicity: 870 singlet, 119 doublet, 11 triplet.

  | Nc | Mai | new (N-PAH) | total |
  | --- | --- | --- | --- |
  | 10–19 | 76 | 9 (0) | 85 |
  | 20–29 | 304 | 49 (1) | 353 |
  | 30–39 | 876 | 181 (9) | 1,057 |
  | 40–59 | 111 | 315 (36) | 426 |
  | 60–79 | 142 | 188 (0) | 330 |
  | 80–99 | 79 | 167 (54) | 246 |
  | 100–129 | 42 | 56 (0) | 98 |
  | 130–216 | 70 | 35 (0) | 105 |

  - Remaining gaps: every in-range v4 species above Nc 100 is already used (91), and hydrogenation and
    pentagon variants are nearly absent from in-range v4 neutrals (17 superhydrogenated, 2 pentagon). Filling
    them needs structures from outside PAHdb with their own geometries (later; not on this plan).
  - Cost at 300 + 600 K × 4 replicas: 83,250 atoms × 8 × 1.22 s ≈ 226 GPU-h, plus ~120 units × ~250 s ≈ 8 GPU-h:
    **~235 GPU-h, ~59 node-hours.**

**2. Domain screen** (driver change, so a new job: one small test job first).
- NNFF forces at the v4 geometries, compared with the same statistic for Mai's species at theirs.
- A short 600 K run (about 10 ps NVT + 5 ps NVE) with the driver's flags (non-finite, T > 30,000 K, > 25
  neighbours, edge overflow) plus a new bond-topology check at the end (no C–H or C–C bond broken or formed).
- Criteria to match the high-T stability test on the project node. Species that fail are dropped and listed,
  not tuned; the selection refills their cells from the next candidates.
- **Implemented without a driver change (2026-11-11):** the screen is an ordinary `mlmd_run.py` plan (600 K,
  4 replicas, 20 ps NVT + 10 ps NVE) and `d1/screen_check.py` compares the bond topology in each unit's last
  checkpoint (`state.npz`, `p`) with the starting geometry, besides the driver's QA flags. An H fails if its
  nearest heavy atom changed; heavy-atom bonds fail beyond 2.0 Å, new ones below 1.6 Å. The force check at the
  start geometry is dropped: the run itself is the test. The same check runs on every production output too.
- **First result, on the calibration run (Mai's species, 2 ps + 2 ps):** 31 of 33 species pass. C28H16_2170
  (2 of 4 replicas at 600 K) and C28H16_2173 (1 of 4) show 1,2-H shifts: an H leaves its carbon (1.08 → 1.4–1.6
  Å) and binds the neighbouring one (2.45 → 1.1–1.2 Å), two H per event, within 4 ps. Both are non-planar
  (z-range 2.3–2.6 Å). A 1,2-H shift in a PAH has a barrier of several eV, so at 600 K this is a model artifact
  and Mai's published 600 K spectra of such species may contain it. Nothing failed at 50 or 300 K.
- **Full screens (jobs 1259237, 1259238; checked 2026-11-12; `runs/screen_*_v1/screen.csv`).** A species fails
  if any of its 4 replicas fails. "Planar" means a thickness (extent along the smallest principal axis of the
  start geometry) below 0.3 Å.

  | Set | Species failing | Planar | Non-planar | Odd H (radicals, no N) | Worst cause: H shift only / bonds / hot or non-finite |
  | --- | --- | --- | --- | --- | --- |
  | New (PAHdb v4.00) | 139 / 1,000 (14 %) | 7 / 786 (1 %) | 132 / 214 (62 %) | 42 / 72 (58 %) | 120 / 9 / 10 |
  | Mai's 1,704 | 365 / 1,704 (21 %) | 9 / 838 (1 %) | 356 / 866 (41 %) | 157 / 565 (28 %) | 334 / 29 / 2 |

  - The failures are almost all non-planar species, mostly the C29–C34 families (C33H19, C34H20, C32H18,
    C31H17, C30H18, C29H17), with 1–4 H moved per replica. N-PAH: 7 of 100 new and 4 of 40 of Mai's fail.
  - The 10 new species that ran hot or non-finite include the N-PAH C52H16–18N2 (`v4u9326, 9281, 9253`), and
    C57H22_v4u808, C59H24_v4u810; C93H21N3 (`v4u8868, 8883`) and C94H22N2_v4u8306 form many bonds.
  - The screen is 30 ps per replica; a 400 ps production run would show more events, so the pass rate is an
    upper bound.
  - Not yet checked by eye: some single "H moved (1)" may be the nearest-heavy-atom test switching for an H
    in a crowded fjord or CH2 site without a real hop, and a few "bonds broken (1)" in small sp3 species
    (C9H8_495, C9H7_493) may be a long C–C bond crossing 2.0 Å. Look at a sample of trajectories before
    quoting the counts.
  - Implication: for non-planar species Mai's model is out of domain at 600 K, and their published 600 K
    spectra for these species are suspect. Dropping and refilling as planned would remove ~62 % of the
    non-planar new species, so the new set would become almost all planar; options: drop and refill, keep them at
    300 K only, or wait for the high-T test. **Decision (user, 2026-11-12): hold** until the high-T test on the
    project node; steps 2's refill and 4 wait.
- **Overlap run screened too (job 1259236, 200 + 200 ps, 8 replicas; `runs/overlap_v1/screen.csv`):** 4 of 48
  species fail (all 4 non-planar; 18 of the 48 are non-planar).
  - C27H15_2249 and C32H18_1655 fail only at 600 K (H shifts, two replicas of C32H18_1655 hot with ~38 bonds
    broken). C32H18_1655 passed the 30 ps screen, which confirms that the screen's pass rate is an upper bound.
  - **C19H12_348 and C17H14_342 also fail at 300 K.** C19H12_348 opens the same sp3 C–C single bond (atoms 18–21,
    CH2–CH, 1.52 Å at the start, 2.7–4.3 Å at the end) in all 8 replicas at 300 K; C17H14_342 opens CH2–CH2 /
    CH–CH2 bonds of its hydrogenated ring (1.52–1.54 → 2.4–2.9 Å) in 3 of 8. None of these start-geometry bonds is
    near the 2.0 Å threshold, so these are real ring openings in the model. Breaking a C–C single bond takes
    ~3.5–4 eV, so at 300 K this is a model artifact for sp3 (superhydrogenated) sites, and Mai's published 300 K
    spectra of such species may contain it. The 600 K screen did not test 300 K, so other sp3 species may do
    the same; a 300 K screen of Mai's sp3 species would show how many.
- **300 K screen of Mai's 79 sp3 species (job 1265105, 200 + 200 ps, 4 replicas; checked 2026-11-13;
  `runs/screen300_sp3_v1/screen.csv`): 22 of 79 fail (28 %) at 300 K.**
  - Worst event per species: 13 lose an H (it dissociates: C16H12_337 r0 and C96H25_688 r1 end with the H
    2,000–4,700 Å away), 5 break a C–C single bond at a CH2 site (C9H8_495, C15H10_341, C17H14_342, C19H12_348,
    C19H14_349; e.g. 1.55 → 4.87 Å), 4 move an H to another carbon (e.g. C10H9_376 CH2 → neighbouring C).
  - The C96H25/C96H26 isomers (684–691): an H moves or leaves in nearly every replica.
  - 21 of the 22 also failed the 600 K screen; C23H13N_232 fails only here (an H lost).
  - Checked by eye on six cases: the flags are real events, not artifacts of the test.
  - Mai's published 300 K spectra of these 22 species are suspect too. For the library: sp3 (superhydrogenated
    or CH2-bearing) species are outside the model's domain even at 300 K about a quarter of the time.
- **`screen_check.py` now reports "H lost"** (an H more than 2.0 Å from every heavy atom; before, a lost H
  could keep its starting carbon as the nearest heavy atom and pass, as C23H13N_232 did). Rerun on all runs
  2026-11-13: pass/fail counts unchanged (overlap 4/48, new 139/1,000, Mai 365/1,704); species with H lost:
  3, 18 and 32.
- **The 18 new sp3 species at 300 K (job 1270903, 200 + 200 ps, 4 replicas; checked 2026-11-13;
  `runs/new300sp3_v1/screen.csv`): 14 of 18 fail; kept: C14H16_v4u339, C27H14_v4u69, C55H20_v4u318,
  C96H28_v4u190.** The heavily hydrogenated small ones (C10H16, C10H18, C18H24; H/C 1.3–1.8, nearly or fully
  saturated) fall apart in every replica (hot, 5–16 H lost, up to 22 bonds broken); the larger ones lose or move
  1–5 H or break one C–C bond. Worse than Mai's sp3 set (22/79), as expected: v4.00's superhydrogenated species
  go further beyond the model's training set (350 superhydrogenated configurations of 17,175). The 14 failures
  are dropped from the 300 K library (user, 2026-11-13).
- **300 K production of the 982 non-sp3 new species (jobs 1268834, 1274456, 1274694; 200 + 200 ps, 4 replicas;
  checked 2026-11-14; `runs/new300_v1/screen.csv`): 62 of 982 fail (143 of 3,928 systems); 920 pass.**
  - Non-planar 56/196 (29 %), planar 6/786 (the N-PAH C52H16–17N2 and C93H21N3/C94H22N2, which also blow up:
    up to 89 bonds broken in one replica).
  - 56 of the 62 show only "H moved (1–2)", mostly in the C29–C34 non-planar families. Checked in four final
    geometries: two H swap between carbons 2.4–2.5 Å apart (each ends 1.13–1.19 Å from the other carbon and
    1.5 Å from its own), or one H sits bridged between them (1.34–1.46 Å to both). Real model artifacts, not
    test artifacts; this was not seen in the 16-ish non-planar overlap species at 300 K.
  - 59 of the 62 also failed the 30 ps 600 K screen; only C26H16_v4u2266, C29H17_v4u2142 and C34H20_v4u984 fail
    here alone. So the 600 K screen predicts 300 K trouble well; 64 species that failed the 600 K screen pass at
    300 K over 400 ps.
  - The 62 are dropped from the 300 K library, as for the sp3 species. 300 K library from the new set: 920 + 4
    sp3 = 924 species.
  - **Dipole check on the whole production (2026-11-14; `spectra/new300_v1_qm0/qa.csv`, rotation-aware version of
    `mlmd_spectra.py`).** Flagged: 57 of 141 failing systems, 11 of 105 passing replicas of failing species, 24 of
    3,680 systems of kept species, in 9 species:
    - rotation false alarms (weakly polar N-PAH, |mu| 0.21–0.23 just under the 0.25 polar cut; every replica at
      vec 0.20–0.26 with mag ≤ 0.03, like C52H17_835 in the overlap run): C31H14N_v4u9356, C52H18N2_v4u9240,
      C53H18N_v4u9214, C53H18N_v4u9225, C95H24N_v4u7524. Keep.
    - likely hidden hops (one or two replicas, C29–C34 families): C30H18_v4u1936 r2 (vec 0.32, mag 0.26),
      C34H20_v4u948 r3 (vec 0.22, mag 0.14), C34H20_v4u964 r0 and r3, C30H18_v4u1946 r2 (vec 0.20–0.21).
      Recommended: drop (the borderline pair of the earlier check is among them).
    - kept sp3 species (`spectra/new300sp3_v1/qa.csv`): C14H16_v4u339 (mag 0.48–0.72) and C55H20_v4u318 (mag
      0.15–0.17) are flagged in every replica; C27H14_v4u69 and C96H28_v4u190 are not. A shift in every replica
      may be conformer interconversion of the saturated ring rather than chemistry; the dipole cannot tell.
      Given 14 of 18 sp3 species fail the screen, recommended: drop both.
    - With these drops the new 300 K set is 916 + 2 sp3 = 918 species (user to confirm).
  - **Final 300 K spectra and library (2026-11-14, user's go).** Made with the merged `mlmd_spectra.py` in the
    production convention (qm=3, no clip, replica NVE temperature; committee review B1) with `--xyz` (bond-graph
    skip, `mlmd_integrity.py`): `spectra/new300_v1` (949 species with ≥ 1 clean replica) and `spectra/new300sp3_v1`.
    The Mai-convention copies are `spectra/new300_v1_qm0` and `new300sp3_v1_qm0`; 33 species' files from the run
    without `--xyz` were set aside in `spectra/new300_v1_stale_noxyz`. The bond-graph check skipped one replica of
    C96H28_v4u190 that the screen passed, so that species is dropped too. **Library list:
    `d1/new300_library_v1.csv`, 917 species** (916 non-sp3 + C27H14_v4u69), each with 4
    clean replicas at 300 K.
  - **Hidden events (dipole check, 2026-11-14).** The screen sees only the final geometry, so an H that hops and
    returns would pass. The driver keeps no trajectory, but `dipole.npy` (NVE, every 1 fs) does: a hop shifts the
    1-ps mean dipole. Metric: largest distance of a 1-ps window mean from the system's median window. On the four
    units with the most failures (n054_u001, n050_u000, n052_u000, n046_u000; C29–C34 families):
    - failing systems: median 0.47 e·Å, 71 of 81 above 0.2 (the other 10 are swaps that leave the dipole nearly
      unchanged, so the metric is a partial check);
    - passing replicas of species that fail elsewhere: 16 of 39 above 0.2 (likely hop-and-return; these
      species are dropped anyway);
    - species with all replicas passing: median 0.044, 99th percentile 0.21, max 0.22 (3 of 157 systems above
      0.2: C34H20_v4u948 r3, C34H20_v4u964 r0 and r3, both of which failed the 600 K screen). No clear hidden
      hops among the kept species in this sample; the three borderline ones are worth a look before use.
    - For future runs (600 K and up), a per-chunk topology check in the driver would catch transient events
      directly.

**3. Overlap run.** 48 of Mai's species, stratified by Nc and class (including a few N-PAH, pentagon and
superhydrogenated ones), × 300 and 600 K × **8 replicas**; 200 ps NVT + 200 ps NVE.
- Each published 300 and 600 K spectrum ranked among our replicas (strong-band centroids, band-window shares, as
  `mai_spectrum_compare.py` does for naphthalene).
- Seed scatter of band shares as a function of Nc; it sets the replica count of step 4 (4 by default; fewer above
  an Nc where the scatter is small).
- ~21 GPU-h (~5–6 node-hours).
- **Result (2026-11-14; `d1/overlap_compare.py`, output `spectra/overlap_v1/compare_mai.csv`).** Spectra with
  `mlmd_spectra.py`; systems failing the screen or the dipole check left out (C19H12_348 at 300 and 600 K,
  C17H14_342, C27H15_2249 and C32H18_1655 at 600 K have fewer than 3 clean replicas and are skipped). Six band
  windows (CH oop 650–950, CC/CH in-plane 1100–1500, CC stretch 1500–1700, CH stretch 2950–3150, 300–650,
  950–1100 cm⁻¹); per window its share of the 300–3800 cm⁻¹ intensity and its centroid.
  - **Our runs reproduce Mai's published spectra within the seed scatter.** The published share lies inside our
    replica range in 70–89 % of species per window (expected 78 % for 8 replicas if both come from the same
    distribution); median share differences −0.008 to +0.002; median centroid offsets ≤ 0.8 cm⁻¹, typical
    |offset| 0.6–3.4 cm⁻¹. The largest systematic is CH oop at 300 K (published share 0.0075 lower than ours,
    inside our range in 72 %).
  - **Replica scatter:** one replica's band share has a relative sd of 7 % (Nc < 30) to 5 % (Nc ≥ 60) at 300 K and
    5.5 % to 3.4 % at 600 K. With 4 replicas the species mean is good to ~2.5–3.5 % at 300 K and ~2–3 % at 600 K;
    2 % at 300 K would need 6–13 replicas. **Recommendation: keep 4 replicas** (step 4's default); the
    regression works on population spectra that average many species, so per-species seed noise averages down
    further.

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

## 900 K pilot (job 1281348, 2026-11-15)

88 of the 857 species (63 planar, 25 non-planar; random within Nc bins), 900 K, 4 replicas, 200 + 200 ps, per-chunk
topology check (screen_check.py criteria, `flags.topo_change_chunk/_chunks`). 47 of 48 units finished (n048_u000,
4 systems, left unfinished); one unit finished before the check fix and is not counted, and units started before the
11:15 cancel were checked only from their resume. Systems with any hop, H loss or broken bond during the run:

| Nc | < 30 | 30–50 | 50–80 | 80–120 | ≥ 120 | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Planar | 4/12 | 3/60 | 6/60 | 11/60 | 6/60 | 30/252 (12 %) |
| Non-planar | 20/20 | 12/16 | 4/20 | 4/20 | 2/12 | 42/88 (48 %) |

- Species with all 4 replicas intact: planar 43/63, non-planar 5/22.
- Of the changed systems, 59 stay changed for most of the run; 13 are brief (< 10 % of the remaining chunks).
- Conclusion: 900 K is workable for planar species with per-replica filtering (~88 % of systems kept); non-planar
  species are mostly lost, the small ones entirely.

## 600 K production (jobs 1280975 cancelled, 1281347; 2026-11-15)

857 species (library species that passed the 600 K screen), 4 replicas, 200 + 200 ps, per-chunk topology check from
the resume (all units were still in NVT when the check was fixed, so the whole NVE part is covered). 93/93 units
done, no errors. Topology changed in 153 of 3,428 systems; 810 species keep all 4 replicas intact.
- Spectra: `spectra/new600_v1` (production convention; systems with a topology change skipped; `qa.csv`): 829
  species with ≥ 1 clean replica.
- Dipole check on the intact systems: 17 flagged, all judged false alarms except C52H18N2_v4u9249 r1 (vec 0.98,
  mag 0.24; that species already lost 2 replicas): the weakly polar N-PAH (C31H14N_v4u9356, C52H18N2_v4u9240,
  C53H18N_v4u9214/9225, C95H24N_v4u7524; vec 0.21–0.24, mag ≤ 0.05) as at 300 K, and the two largest planar
  species (C172H38_v4u6426, C196H38_v4u6418; vec 0.20–0.21, |mu| 0.20), thermal size at 600 K.
- **600 K library: `d1/new600_library_v1.csv`, 810 species with 4 intact replicas.**

## 900 K production (jobs 1282825 and 1284012, two disjoint unit lists; 2026-11-15/16)

779 planar species of the 857, 4 replicas, 200 + 200 ps, per-chunk topology check throughout. 85/85 units, no
errors. Topology changed in 345 of 3,116 systems (11.1 %, as in the pilot's 12 % for planar species); by Nc:
< 30 4/12, 30–50 67/896, 50–80 137/1,304, 80–120 118/780, ≥ 120 19/124. 549 species keep all 4 replicas intact.
- Spectra: `spectra/new900_v1` (production convention, topology-changed systems skipped, `qa.csv`): 760 species.
- Dipole check at 900 K: 24 of 2,771 intact systems above the thresholds (vec 0.20–0.30, mag ≤ 0.21), but they
  are the tail of a continuous distribution (nonpolar vec: median 0.064, 99 % 0.19, 99.9 % 0.28; at 600 K 0.056,
  0.17, 0.22), unlike the hops at 300–600 K (0.4–2.0). With the per-chunk topology check present for the whole run,
  it is the primary filter; the dipole flags stay in `qa.csv` for reference and drop nothing.
- **900 K library: `d1/new900_library_v1.csv`, 549 species with 4 intact replicas.** 549 species are in all
  three libraries (300, 600 and 900 K).

## Band trends with temperature, first pass (2026-11-16; `d1/tdep_fit.py`, `spectra/tdep_bands_v1.csv`)

549 species in all three libraries (planar neutrals), production-convention spectra, x = each replica's NVE mean T
(12 points per band). Bands as in `tdep_compare.py` (6 strongest at 300 K, followed to 600 and 900 K), measured in a
+-20 cm^-1 window; 3,294 bands. Per 100 K, median [10-90 %]:

| Family | Centroid shift | Width (in window) | Share | Significantly curved (centroid) | Quadratic - linear at 1200 K |
| --- | --- | --- | --- | --- | --- |
| CH oop (600-1000) | -1.46 [-2.48..-0.14] | +0.27 | -1.0 % | 54 % | 2.2 cm^-1 |
| CC/CH in-plane (1000-1700) | -3.57 [-5.59..-2.12] | +0.28 | -3.6 % | 56 % | 5.0 cm^-1 |
| CH stretch (2800-3300) | -5.49 [-6.61..-4.38] | +0.32 | -10.7 % | 58 % | 8.5 cm^-1 |

- Shifts depend weakly on size: in-plane -4.0 (Nc < 40) to -3.2 (Nc 90-130); oop -1.7 to -1.4; CH stretch -5.1
  to -6.0 cm^-1 per 100 K.
- **The +-20 cm^-1 window underestimates broad bands.** Over whole regions the CH stretch shifts by -9.9 [-10.2..-9.6]
  and broadens by +6.6 cm^-1 per 100 K (matching `charged_tdep_validation.md`: -10.6 to -11.8 for small species),
  the oop region (800-950) by -1.7 and +0.6. Band widths from the window method are not usable; a profile fit per
  band (e.g. Drude, as PAHFIT measures them) is needed before the trends become rules.
- **Linearity:** the curvature is often statistically significant (small errors from 12 points), but small in size:
  extrapolating linearly to 1200 K differs from the quadratic by 2-8 cm^-1 per band (region-level: oop ~0, CC +2,
  CH stretch +4.9 against a total shift of ~-90 cm^-1), i.e. ~5-15 % of the shift. Shifts accelerate slightly with T
  for oop and in-plane bands. A linear extrapolation to 1200 K looks defensible for positions with that uncertainty;
  widths and shares need the profile fits first.

## Emission temperatures: how much IR comes out above 900 K? (2026-11-16; `d1/emission_temp.py`, `spectra/emission_temp_v1.csv`)

Thermal approximation (as PAHdb's cascade): after absorbing a photon of energy E, U(T_max) = E with U from the
species' PAHdb v4.00 harmonic modes (all 3N-6); cooling is by IR alone, so the emitted energy fraction above T_c is
1 - U(T_c)/E. 917 library species (neutral). Median share of emitted IR energy above 900 K [median T_max]:

| Nc | 3 eV | 6 eV | 8 eV | 10 eV | 13.6 eV |
| --- | --- | --- | --- | --- | --- |
| < 30 (32) | 0 [731 K] | 0.27 [1087] | 0.45 [1300] | 0.56 [1504] | 0.68 [1858] |
| 30-50 (291) | 0 [636] | 0.05 [930] | 0.29 [1104] | 0.43 [1270] | 0.58 [1555] |
| 50-80 (343) | 0 [529] | 0 [759] | 0 [892] | 0.19 [1018] | 0.40 [1233] |
| 80-120 (200) | 0 | 0 | 0 | 0 [791] | 0.08 [945] |
| >= 120 (51) | 0 | 0 | 0 | 0 | 0 [776] |

Above 1200 K: only Nc < 50 with E >= 8 eV (up to 0.48 for Nc < 30 at 13.6 eV, 0.33 for Nc 30-50). Above 600 K:
most of the emission for Nc < 80 at E >= 6 eV.
- **Reading:** temperatures above 900 K matter only for small PAHs (Nc < ~80) in hard fields (E >= 8 eV), but there
  they carry a large share (20-70 %) of the emission. D1 draws Nc means of 30-120 and E of 5-13.6 eV, so a
  substantial part of the population grid depends on the > 900 K trends. These are energy fractions over all bands;
  short-wavelength bands (3.3 um) are emitted preferentially at high T, so their > 900 K share is larger still.
- The 300 and 600 K MLMD set nearly all emission for Nc >= 80 at any E, and for all sizes at E <= 6 eV (apart from
  Nc < 30).

## Band profiles and trends, 300-900 K (2026-11-16; `d1/profile_calib.py`, `d1/profile_fit.py`)

549 planar neutral species in all three libraries; production-convention spectra; every replica fitted against its
own NVE mean T (12 points per band). Fits run on the login node in 3 chunks of 183 species (6-7 min each);
outputs `spectra/profile_fits_v1_c{0,1,2}.csv`.

- **Calibration (`profile_calib.py`):** synthetic Lorentzian lines through `ir_sigma` at production settings (ACF
  truncated at 16 ps, an instrumental width of ~2 cm^-1 with ringing). A truncated-Lorentzian profile recovers FWHM
  from 0.5 to 32 cm^-1 (scatter 10-15 % per band with 4 replicas); a plain Drude fails below ~1.5 cm^-1.
- **Blends are measured as complexes** (fixed windows 3.3: 2850-3250, 6.2: 1540-1660, 7.7: 1220-1470, 8.6:
  1100-1220 cm^-1): multi-component fits there are degenerate (20-species test). Validation gate: 3.3 um centroid
  -9.1 and broadening +5.3 (rms) / +6.8 (FWHM) cm^-1 per 100 K, matching the region-level -9.9 and +6.6.
- **Isolated out-of-plane bands** are profile-fitted (components overlapping a neighbour within one FWHM at 900 K
  are left out of the trends).

Per 100 K, median over species [10-90 %]; share = band or window integral relative to its 300 K value:

| Band | Position | Width | Strength | FWHM at 300 K |
| --- | --- | --- | --- | --- |
| 3.3 um (complex) | centroid -9.1 [-9.4..-8.8]; peak -5.8 | rms +5.3; FWHM +6.8 [+5.4..+8.0] | -4.4 % | 20.0 |
| 6.2 um (complex) | peak -4.2 [-7.2..-0.6] | rms +0.85 | -0.4 % | 16.5 |
| 7.7 um (complex) | peak -3.3 [-9.1..+4.7] | rms +0.70 | -1.1 % | 10.0 |
| 8.6 um (complex) | peak -1.6 | rms +0.41 | +3.0 % | 9.8 |
| 11.2 um (421 isolated) | -1.71 [-2.56..-0.67] | FWHM +0.85 [-0.38..+1.50] | area -4.3 %, window -4.7 % | 3.8 |
| 12.0 um (361) | -1.67 | +0.74 | -1.3 % | 3.6 |
| 12.7 um (465) | -0.89 | +0.55 | -2.9 % | 3.6 |
| 13.5 um (551) | -0.84 | +0.67 | -1.6 % | 3.3 |

- Fixed-window centroids of the 6-9 um complexes stay near 0 because broadening and spill-over balance the shift;
  their peaks carry the shift. Directly read FWHMs of these complexes jump when sub-peaks merge (quadratic-linear at
  1200 K of 15-28 cm^-1): use rms widths for them.
- **Linearity for 1200 K:** quadratic minus linear at 1200 K is 1-3 cm^-1 for the oop band positions and widths;
  for the 3.3 um band +7 (centroid; the shift slows) and -6 to +8 (width) against a total change of ~-80 and +50
  cm^-1, i.e. ~10-15 %. Band strengths are linear (|q-l| <= 0.1).
- **For the diagnostics:** the 11.2 um band loses ~4-5 % of its strength per 100 K relative to 300 K while the
  6.2 um complex stays flat, so 6.2/11.2 in these spectra rises by ~4-5 % per 100 K (~25 % from 300 to 900 K) at
  fixed charge. Emission spectra weight temperatures by the cascade; this is the input it needs, not the emission
  ratio itself.
- Not yet done: comparison with laboratory hot-PAH spectra; size dependence of these trends (all numbers above are
  medians over Nc 22-216).

### Size dependence (`profile_fit.py bysize`)

Median per Nc bin (Nc 20-40 / 40-60 / 60-90 / 90-130 / 130-220; 55 / 217 / 168 / 98 / 11 species):
- **Positions.** 3.3 um: -9.1 in every bin (no size dependence). 6.2 um: -4.7 / -4.1 / -3.9 / -4.5 / -5.1 (none
  significant). 7.7 um: -3.8 to -2.8, weakly smaller for large ones. Out-of-plane bands shift more in small PAHs:
  11.2 um -2.3 to -1.6, 12.7 um -1.2 to -0.9, 13.5 um -1.4 to +0.6 (+2.4 +- 0.2 per decade of Nc; band identity by
  position, so in large PAHs the 13.5 um window holds different modes).
- **Broadening.** 3.3 um +6.5 to +7.6 (+1.1 +- 0.3 per decade); 6.2 um +0.8 to +1.6 (+1.2 +- 0.2 per decade);
  others flat within errors (oop +0.4 to +0.9).
- **Strength.** 3.3 um -4.0 to -4.8 % per 100 K; 6.2 um ~0; 7.7 um -1 to -1.6 %; 8.6 um +2.6 to +3.3 %; 11.2 um -5.8 %
  (Nc 20-40) to -3.9 %.
- **Widths at 300 K** fall with size for the oop bands (FWHM 4.2-4.6 cm^-1 at Nc 20-40 to 2.1-2.9 at Nc >= 130) and for
  3.3 um (23.8 to ~19-20).
- For rules: per band a constant temperature slope is adequate except the oop band positions (and the 300 K oop
  widths), which need a log10(Nc) term.

## Temperature rules, version 1 (2026-11-16; `d1/tdep_rules_v1.csv` from `profile_fit.py rules`)

Planar neutral PAHs, Nc 22-216, 300-900 K (replica NVE temperatures), Mai et al.'s model, production spectral
convention (qm=3, unclipped, replica T), 549 species. Each entry: value at Nc = 50 ± scatter between species
(1.4826 MAD); a size term (per decade of Nc, i.e. per factor of 10 in Nc) is shown only where it exceeds 2 s.e.
Strength is relative to the band's own 300 K value. Last column: quadratic minus linear fit at 1200 K (position /
width), the extrapolation uncertainty.

| Band (um) | Measured as | Position shift (cm^-1 per 100 K) | Broadening (cm^-1 per 100 K) | Strength per 100 K | Width at 300 K (cm^-1) | quad-lin at 1200 K |
| --- | --- | --- | --- | --- | --- | --- |
| 3.3 | complex | -9.11 ± 0.24 | +6.66 ± 0.98 (+1.05/dex) | -4.3 % ± 0.3 (-1.1 %/dex) | +21.5 ± 3.4 (-5.3/dex) | +7.1 / +7.9 |
| 6.2 | complex | -4.01 ± 2.44 | +0.72 ± 0.58 (+1.17/dex) | -0.4 % ± 1.8 (+2.6 %/dex) | +29.1 ± 2.8 (-4.5/dex) | +4.9 / +0.1 |
| 7.7 | complex | -2.81 ± 3.49 | +0.73 ± 0.51 | -0.9 % ± 1.0 (-1.3 %/dex) | +63.0 ± 4.0 (-3.3/dex) | -4.0 / -0.3 |
| 8.6 | complex | -0.47 ± 2.12 (-3.43/dex) | +0.39 ± 0.40 | +3.1 % ± 1.8 | +29.2 ± 3.3 | -4.1 / +0.7 |
| 11.2 | isolated oop band | -1.90 ± 0.61 | +0.67 ± 0.64 | -3.8 % ± 6.8 | +4.6 ± 1.3 (-3.2/dex) | -1.4 / -0.9 |
| 12.0 | isolated oop band | -1.64 ± 1.35 | +0.64 ± 0.89 (+0.70/dex) | +2.6 % ± 11.0 | +4.5 ± 1.3 (-4.9/dex) | -0.5 / -0.3 |
| 12.7 | isolated oop band | -0.97 ± 0.89 (+1.39/dex) | +0.52 ± 0.73 | -0.1 % ± 10.3 (+18.7 %/dex) | +5.0 ± 1.5 (-4.1/dex) | -2.9 / -0.6 |
| 13.5 | isolated oop band | -1.03 ± 0.48 (+2.36/dex) | +0.75 ± 0.46 (-0.66/dex) | -1.0 % ± 7.0 | +4.2 ± 1.1 (-4.9/dex) | -2.7 / +0.2 |

How to read and use it:
- **Widths are of two kinds.** For the isolated oop bands (11.2-13.5 um) the width is the FWHM of one band, i.e. a
  per-line width (~4.5 cm^-1 at 300 K, Nc = 50, growing ~0.5-0.75 cm^-1 per 100 K). For 3.3 um it is the FWHM of the
  blended CH-stretch feature, and for 6.2/7.7/8.6 um the rms width of the complex in its window; these include the
  spread of the underlying harmonic lines and are not per-line widths.
- **For the emission model**, which broadens harmonic stick spectra: shift each line by its family's position rule
  (CH stretch from 3.3; CC stretch from 6.2; CC/CH in-plane from 7.7 and 8.6; oop from 11.2-13.5 by position) and give
  it the per-line width from the oop rule as a first choice; then check that the broadened sticks reproduce the
  complex widths in this table (a closure test still to do).
- **Strengths**: 3.3 um -4.3 % and 11.2 um ~-4 % per 100 K against 6.2 um ~0 and 8.6 um +3 %; per-band oop strengths
  scatter by ~7-11 % per 100 K between species, so use them as population means.
- **Validity**: neutral, planar, Nc 22-216, 300-900 K. Extrapolation to 1200 K adds the last column (1-3 cm^-1 for oop
  positions, 4-7 for the 3.3/6.2/7.7/8.6 positions, up to 8 for the 3.3 um width). Non-planar, superhydrogenated and
  charged species are outside it (charged: positions and widths transfer within a few cm^-1 per
  `charged_tdep_validation.md`; strengths unsettled).

### Closure test: rules on PAHdb harmonic sticks (2026-11-16; `d1/rules_closure.py`, 60 random species of the 549)

PAHdb v4.00 sticks shifted by the family position rules and broadened with Lorentzians, against the MLMD feature
widths (same windows and measures). Only widths and their growth are compared (PAHdb, B3LYP/6-31G* scaled 0.9794,
and Mai's model put lines at different absolute positions).
- **The oop per-line width fails for the other families.** With FWHM 4.5 cm^-1 + 0.67 per 100 K on every line, the
  3.3 um feature is 9 / 12 / 15 cm^-1 at 300 / 600 / 900 K against 19 / 37 / 60 in MLMD; the 6.2 and 7.7 um complexes do
  not broaden at all (MLMD +5 and +4 cm^-1 rms). CH-stretch lines broaden much faster than oop lines.
- **Per-family per-line widths by inversion** (the FWHM for which the broadened sticks reproduce the MLMD feature
  width; `spectra/perline_width_v1.csv`):
  - **CH stretch (3.3 um): 9.9 [7.1-15.3] / 24.0 [17.2-28.1] / 44.4 [33.9-53.7] cm^-1 at 300 / 600 / 900 K, +5.8 per
    100 K, no size dependence (+0.2 per decade), a solution for all 60 species.** Usable as the CH-stretch rule.
  - 6.2 / 7.7 / 8.6 um: ill-posed. Implausible widths (6.2 um: 45 -> 142 cm^-1) and no solution for many species
    (7.7 um: 19-34 of 60). The window rms width saturates as Lorentzian wings fill the window, and PAHdb and Mai's
    model spread the CC lines differently already at 300 K, so the inversion fits that model difference.
- **Per-line widths, status:** oop lines 4.5 + 0.5-0.75 per 100 K (direct fits); CH stretch 9.9 + 5.8 per 100 K
  (inversion); CC stretch and in-plane bends not determined. Options: profile-fit the isolated in-plane bands in the
  MLMD spectra (a minority of bands, as in the first prototype); or invert on band shapes rather than window rms
  (e.g. match the MLMD complex profile after aligning positions); or, as an interim rule, use the oop growth rate
  with an uncertainty spanning the CH-stretch value.

### Per-line widths by vibration family (2026-11-16; `d1/perline_widths_v1.csv`)

In-plane region (1100-1700 cm^-1) profile-fitted for 200 random species of the 549 (`profile_fit.py fit --regions
inplane --sample 200`, 2 login-node chunks of ~13 min; `spectra/inplane_fits_v1_c{0,1}.csv`); bands kept if isolated
at 300, 600 and 900 K (nearest component farther than 0.75 x the sum of the FWHMs, FWHM off the fit bounds, >= 0.3 % of
the total at 300 K; `d1/inplane_widths.py`). The fits are partly degenerate at 900 K (components collapse or swell),
so most components are rejected (290 of 2,398 kept).

| Family | Range (cm^-1) | Per-line FWHM at 300 K | Broadening per 100 K | Bands | Source |
| --- | --- | --- | --- | --- | --- |
| CH stretch (3.3 um) | 2800-3300 | 9.9 [7.1-15.3] | +5.8 | 60 species | inversion of the 3.3 um feature on PAHdb sticks |
| CC stretch (6.2 um) | 1540-1700 | 14.3 [10.7-22.1] | +3.7 [+1.6..+4.4] | 14 (small sample) | isolated in-plane bands |
| 1470-1540 | | 7.4 | +2.1 | 37 | isolated in-plane bands |
| CC/CH in-plane (7.7 um) | 1220-1470 | 5.9 [3.9-11.0] | +1.75 [+0.5..+2.9] | 145 | isolated in-plane bands |
| CH in-plane (8.6 um) | 1100-1220 | 6.0 [4.1-10.3] | +2.5 [+1.5..+4.2] | 90 | isolated in-plane bands |
| CH out-of-plane | 650-1000 | 4.5 | +0.5-0.75 | ~1,800 | isolated oop bands |

- Broadening rises with vibrational frequency (oop +0.6, in-plane +1.7-2.5, CC stretch +3.7, CH stretch +5.8 cm^-1 per
  100 K), as expected from anharmonicity. No significant size dependence except CH in-plane (+1.4 per decade of Nc).
- **Closure with these widths** (60 species, PAHdb sticks shifted by the position rules): feature width growth
  300 -> 900 K, rules vs MLMD: 3.3 um +39.5 vs +40.4 (absolute 20/42/60 vs 19/37/60 cm^-1), 8.6 um +2.9 vs +2.6,
  6.2 um +2.9 vs +5.3, 7.7 um -0.1 vs +4.2. The remaining 6.2/7.7 um growth in MLMD is likely the broad background that
  grows with T in the in-plane region (also seen in `charged_tdep_validation.md`), which per-line Lorentzians do not
  produce: it would need a separate pedestal term in the emission model.
- Caveat: isolated bands are a minority and may be the less congested ones; the 6.2 um value rests on 14 bands.

### Pedestal (broad background) against temperature (2026-11-16; `d1/pedestal.py`)

Lower envelope of each spectrum (running minimum over 41 cm^-1, then a running mean), integrated per region as a share
of the 300-3800 cm^-1 total. MLMD, 549 species, median at 300 / 600 / 900 K and per-species slope:

| Region (cm^-1) | 300 K | 600 K | 900 K | Slope per 100 K |
| --- | --- | --- | --- | --- |
| oop 650-1000 | 0.9 % | 3.5 % | 5.4 % | +0.82 % |
| in-plane 1100-1700 | 4.0 % | 6.9 % | 8.3 % | +0.73 % |
| gap 1700-2800 (overtones, combinations) | 1.6 % | 4.1 % | 6.9 % | +0.92 % |
| CH stretch 2800-3300 | 8.3 % | 13.2 % | 14.6 % | +1.05 % |

The same floor measured on PAHdb sticks broadened with the per-line widths (60 species) gives the part that band wings
explain. MLMD minus sticks, at 300 / 600 / 900 K: oop -1.4 / -0.3 / +0.6 %; in-plane -8.5 / -10.3 / -11.0 %; **gap +2.2 /
+5.1 / +8.4 %**; CH stretch +3.2 / -0.3 / -2.6 %.
- **In-plane: no pedestal beyond the band wings.** MLMD's floor there is lower than the broadened sticks' (its
  in-plane intensity is lower overall than PAHdb's) and grows less. So the hypothesis was wrong: adding a pedestal
  changes nothing, and the 6.2/7.7 um rms-growth gap of the closure (+2.9 vs +5.3, -0.1 vs +4.2 cm^-1) remains. Likely
  causes: the congested 6.2/7.7 um lines broaden faster than the isolated ones the per-line widths come from, and
  PAHdb and Mai's model spread these lines differently.
- **Gap 1700-2800 cm^-1: a real, growing background** that harmonic sticks lack: overtone and combination intensity
  (the region of the observed 5.25 and 5.7 um bands), rising from ~2 % to ~8 % of the total between 300 and 900 K
  (+1 % per 100 K). The emission model needs it as a separate component if the 5-6 um region matters.

### Issue figures and a PAHdb intensity anomaly (2026-11-16; `d1/issue_plots.py`, PNGs in `/appsdata/t641a02/d1/figs/`)

Six figures: spectra MD vs rules for one species; width growth rules vs MD (60 species); background by region; emission
share above 900 K; model breakdown by T and planarity; PAHdb intensities.
- **PAHdb anomaly:** 164 of the 917 library species (18 %; 115 of the 549 in all three temperature libraries) have PAHdb
  v4.00 harmonic spectra dominated by 6-9 um: total intensity up to 1.5e5 km/mol (median of all 1,442), CH-stretch share
  < 15 % (median 42 %), most recorded as closed-shell singlets. The share falls continuously with total intensity, not
  as a separate cluster. Mai's dipole model gives these species ordinary neutral spectra (CH-stretch share 40-50 %).
  Likely low-gap (diradicaloid) structures where B3LYP harmonic intensities are unreliable; this matters for D1's
  harmonic library itself, not only for the rules. Not investigated further yet.
- The closure and pedestal comparisons repeated without these species (CH share >= 0.15) give the same conclusions:
  width growth 3.3 um +42.0 vs +39.3, 8.6 um +2.8 vs +2.4, 6.2 um +1.9 vs +5.2, 7.7 um +0.3 vs +3.6 cm^-1; floor excess
  in the 1700-2800 cm^-1 gap +2.0 / +4.9 / +8.0 %, in-plane -4.5 / -5.1 / -5.0 % (was -8.5 to -11 % with them). The
  figures use only these species.

## PAHdb intensity anomaly: non-Kekule species computed as singlets (2026-11-16; `d1/pahdb_intensity_survey.py`, `d1/kekule_check.py`)

- **Survey of all 10,718 PAHdb v4.00 species** (`spectra/pahdb_intensity_survey.csv`): among pure C/H neutrals without
  sp3 C, 245 of 3,028 (8 %) have a CH-stretch share < 15 % and ~6x the per-carbon intensity of normal neutrals (median
  144 vs 24 km/mol per C; strongest mode median 778 km/mol near 1170 cm^-1). 244 of the 245 are listed as closed-shell
  singlets; doublets and triplets are normal. They cluster at Nc 50-120 (17-22 % there) and at uid >= ~9,700 (v4.00
  additions); none has a cation in PAHdb, and 98 % have a normal isomer of the same formula.
- **Cause: no Kekule structure.** All 239 anomalous pure-C/H closed-shell neutrals have an sp2 framework with no
  perfect matching, sublattice imbalance 2 (one case 4). Such molecules cannot be closed-shell; by Lieb's theorem an
  imbalance of 2 means a triplet ground state. PAHdb lists them as singlets; the intensity pattern is what a
  closed-shell calculation of a diradical would give (whether restricted or broken-symmetry is not recorded).
- **Class-level confirmation** (pure C/H neutrals): non-Kekule, |imbalance| 2, listed as triplets: 51 species, median
  23 km/mol per C, CH share 0.43 (normal, like Kekule species: 23.5, 0.41); the same class listed as singlets: 312
  species, 117 km/mol per C, CH share 0.11. No molecule is listed in both states (formula + bond-graph hash), so there
  is no same-molecule comparison.
- **Extent, pure C/H only** (N changes the electron count, so N-PAH are not tested by this check): 334 PAHdb v4.00
  neutrals are non-Kekule and listed as singlets (|imbalance| 2: 312, 4: 21, 6: 1); 239 fall below the 15 % CH share,
  but those above it are still 2.7x the Kekule per-carbon intensity (64 vs 23.5 km/mol), so the Kekule flag, not the
  CH-share cut, is the exclusion criterion. In our libraries: 184 of 917 (300 K), 183 of 810 (600 K), 137 of 549
  (900 K); N-PAH (untested): 94, 80, 31. Odd-H radicals (doublets) also lack a Kekule structure but are listed and
  computed as open-shell and look normal (721 species, 23.9 km/mol per C, CH 0.42).
- **MD temperature trends are the same for these species** (144 of the 549): e.g. 3.3 um shift -9.04 vs -9.12, width
  +6.66 vs +6.88 cm^-1 per 100 K; 11.2 um -1.56 vs -1.75; all bands within the scatter. So the temperature rules stand;
  the problem is the PAHdb harmonic spectra (and, for MD, that Mai's model treats them closed-shell like the rest).
- The CH-stretch per-line width redone without non-Kekule singlets (60 of the 405 remaining species): 10.6 cm^-1 at
  300 K, +5.86 per 100 K (was 9.9, +5.8); the rule stands.
- **For D1's harmonic library**, decide: exclude the 334 pure C/H non-Kekule "singlets" of PAHdb v4.00 (and their
  counterparts in the surrogate's training data, which is PAHdb), or recompute them in the high-spin state (container D
  DFT pipeline). Mai's model presumably saw such species under the same closed-shell assumption, so identical MD
  temperature trends show consistency, not correct physics; the temperature rules stand either way. Not mentioned in
  the reference corpus; worth reporting to the PAHdb team.

### Comparison with PAHdb v3.20 (2026-11-16)

PAHdb v3.20 (4,233 species, 2019-09-12) downloaded on the login node to `/appsdata/t641a02/d1/pahdb/` (14.2 MB; sha256 in
`t0/pahdb-v3.20.sha256`); the same survey and Kekule check (`spectra/pahdb320_*.csv`). Pure C/H neutrals:

| Class (stated multiplicity) | v3.20: n, km/mol per C, CH share | v4.00: n, km/mol per C, CH share |
| --- | --- | --- |
| Kekule (singlet) | 1,224, 27.9, 0.41 | 1,874, 23.5, 0.41 |
| non-Kekule |imb| 1 (doublet) | 705, 27.5, 0.43 | 721, 23.9, 0.42 |
| non-Kekule |imb| 2 (triplet) | 51, 26.4, 0.43 | 51, 22.9, 0.43 |
| non-Kekule, listed as singlet | **3**, 49-65, 0.25-0.28 | **334**, 117 (|imb| 2), 0.11 |

- v3.20 lists only three non-Kekule species as singlets: C32H18 uid 1741 (|imb| 2), C82H24 uid 561 (4), C126H30 uid 569 (6).
  They are unchanged in v4.00 (same states; 50-65 km/mol per C, CH share 0.22-0.28, i.e. ~2x normal but not extreme).
- **The other 331 are new in v4.00** (uid up to 10,770; v3.20's maximum is 4,287), including every extreme case (CH share
  < 15 %). v3.20 listed its 51 imbalance-2 neutrals as triplets, with normal spectra.
- So the problem is essentially a v4.00 one: the species added in v4.00 that cannot be closed-shell were listed (and
  evidently computed) as closed-shell singlets. Mai et al.'s training data came from v3.2, which contained only the
  three mild cases.

### Decision (user, 2026-11-17): exclude now, test cheap recomputation

- **The 334 pure C/H non-Kekule "singlets" of PAHdb v4.00 are excluded from D1's harmonic library** (list:
  `d1/pahdb_v4_exclude_nonkekule_v1.csv`, from `docs/reports/pahdb_v4_nonkekule/`). They are also in our MD selection:
  184 of the 1,000 new species, 184 / 183 / 137 of the 300 / 600 / 900 K libraries; the temperature rules do not
  depend on them. **The T0b slice contains one: C52H20 uid 10452 (`slice/species.csv`); its neutral/cation pair needs
  replacing.**
- Recomputing them as triplets with finite-difference frequencies would take ~3,400-5,500 H100 GPU-h (median 7-11 h
  per species; C90-130 59 % of it). Next: test analytic frequencies (GPU4PySCF `hessian/uks.py`, available) with IR
  intensities from the CPHF response, which GPU4PySCF implements only for closed-shell (`properties/ir.py`); an
  open-shell version is needed and is validated first on the naphthalene cation (uid 359).

### What the v4.00 paper says (checked 2026-11-17)

Ricca et al. 2026, ApJS 282, 7 (doi 10.3847/1538-4365/ae1c38; open access, read in full): **no mention of electronic
states, spin multiplicity, triplets, open-shell species or Kekule structures**. Section 2.1 gives only the method
(B3LYP/6-31G*, Gaussian 09/16, three scale factors); Section 2.2 says the 4,651 new pure PAHs have "edge defects such as
bay regions" and cites Ricca et al. 2024 (ApJ 968, 128) for them. That paper was not accessible here (full text not
found), so whether it discusses the electronic states of those species is unchecked.
- The paper's Orion Bar demonstration (Sect. 6, Fig. 7, Table 2) fits with a pool that includes these species (pure
  PAHs, charge > -1, C > 20) and reports a large improvement at 6.2 um and 10-15 um over v3.20. Whether the anomalous
  species, with their very strong 6-9 um bands, enter that fit was not checked.

### Open-shell analytic IR: test passed (job 1294269, 2026-11-17, 10:49 on one node; `d1/uks_ir.py`, `d1/uks_ir_test.py`)

B3LYP/6-31G* (6D, (99,590), conv_tol 1e-11) with GPU4PySCF in container D, at PAHdb v4.00 geometries, one H100 each;
results in `/appsdata/t641a02/d1/uks_ir_test/`.
- **Code check, naphthalene (uid 330):** our open-shell routine on a UKS calculation vs GPU4PySCF's closed-shell IR on
  RKS: frequencies within 0.04 cm^-1, intensities within 0.54 km/mol (strong bands within 2 %); both match PAHdb (total
  285.1-285.5 vs 285.5 km/mol, frequencies -0.3 cm^-1 after unscaling, strong-band intensities 0.3 %).
- **Physics check, naphthalene cation (uid 359, doublet):** total 538.7 vs PAHdb 539.1 km/mol, frequencies -0.3 cm^-1,
  strong-band intensities within 0.2 %, <S^2> 0.763.
- **One affected molecule, C30H14 uid 10770 (PAHdb: "1-A1", 9,168 km/mol, CH share 5.7 %):**
  - closed-shell singlet (RKS) reproduces PAHdb's spectrum: 9,333 km/mol, CH share 5.7 %;
  - the triplet (UKS, <S^2> 2.12) is 18.7 kcal/mol lower in energy and has a normal spectrum: 601 km/mol, CH share
    50 %, no imaginary frequencies at the PAHdb geometry.
  This confirms the explanation on a single molecule: the anomaly is the closed-shell singlet treatment.
- **Cost:** C30H14 triplet: SCF 21 s, analytic Hessian + IR 400 s on one H100 (finite differences: ~264 points x
  ~29 s = ~2.1 h, i.e. ~18x more). Scaling the Hessian as nC^2.5-3.5 and adding a 30-step triplet re-optimisation:
  **~700-1,800 GPU-h for all 334** (species with < 90 C: ~210-360 GPU-h), against 3,400-5,500 by finite differences.
  The exponent is not measured yet; one larger species (C90-C100) would pin it. The geometry re-optimisation needs an
  optimiser (none in container D: no pyberny; geomeTRIC import did not complete; GPU4PySCF ships an ASE-based one).

### Triplet scaling test failed (job 1294335, 2026-11-17, 45 min; `d1/triplet_freq.py`, `d1/slurm_triplet_test.sbatch`)

- **C84H24 (6512) and C96H26 (9390):** `CUTENSOR_STATUS_NOT_SUPPORTED` in GPU4PySCF's UHF CPHF (`hessian/uhf.py`
  `_get_jk_mo`, a plain array addition). Container D sets `CUPY_ACCELERATORS=cub,cutensor`, which sends elementwise
  CuPy operations through cuTENSOR; that fails for these array sizes. Fix to test: `CUPY_ACCELERATORS=cub` (GPU4PySCF's
  own tensor contractions keep cuTENSOR).
- **SCF convergence:** the C30H14 optimisation (ASE BFGS through `gpu4pyscf.tools.ase_interface`) stopped at step 14 when
  one SCF did not converge (forces had reached 0.029 eV/A); the C60H20 SCF at the PAHdb geometry did not converge in 200
  cycles, and the driver went on to the Hessian anyway (a bug: it must stop). Fixes to test: start each SCF from the
  previous density, and fall back to second-order (Newton) SCF from the current orbitals, as `d1/aimd_pyscf.py` does
  (commit 64b3aba); stop if still unconverged.
- **Timing that did come through:** C60H20 triplet analytic Hessian + IR 1,761 s (nao 940) vs C30H14 400 s (nao 478):
  cost ~ nao^2.2. Estimate for all 334: ~520 GPU-h (frequencies ~290 + 30-step optimisations ~230); nC < 90: ~180 GPU-h;
  largest species ~19 h. Not yet confirmed above C60.

