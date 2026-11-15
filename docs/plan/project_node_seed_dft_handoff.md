# Handoff: seed-set DFT results back on the project node

Written 2026-11-16 on KISTI Hangang, for the next session on the project node (`gh200`), person or Claude. Branch
`feat-gpu-gh200`. Context: Phase 1 of `docs/plan/charged_mlff_plan.md` (seed set for the charged-state MLFFs); the
Hangang side followed `docs/plan/hangang_seed_dft_handoff.md`. The plan's 2026-11-15 log entry holds the decisions;
this note says what arrived, what to run and what is still open.

## What was done on Hangang

- **Production:** all 7,504 seed frames labelled with GPU4PySCF 1.9.2 (container D), B3LYP/6-31G\*, the settings of
  `d1/dft_seed_run.py`.
  - The check job reproduced 15 pilot frames to ≤ 7e-12 Ha (energy), 4e-12 Ha/bohr (forces), 4e-11 D (dipole).
  - 137 GPU-h, 6.5 h on 3 nodes; every frame converged, 3 needed the Newton fallback (cation MD frames, none with a
    stability test).
  - Hangang runs large frames 1.4–1.9× slower than the project node (two processes per GPU); on its own the GPU is
    ~10–15 % slower. Driver 570.124.06 (CUDA 12.8) runs the image's CUDA 12.9 through minor-version compatibility.
- **QA flags from `collect`:** 50 neutral frames RKS→UKS unstable (10 pairs); 34 cation frames with ⟨S²⟩ > 0.85.
- **Broken-symmetry check** (new opt-in mode `run --stab-all --bs`, needs `--names`): all 124 neutral minima tested.
  - 9 pairs are open-shell at the minimum (E_RKS − E_BS 2.8–22.8 mHa, ⟨S²⟩_BS 0.82–1.12, max |ΔF| 0.35–1.37 eV/Å);
    2 are barely unstable (≤ 0.13 mHa); 113 are stable.
  - C13H10_55 is open-shell at its minimum; 16 of its 32 neutral MD frames have a changed bond graph (an isomer
    ~27 mHa lower), which is why they looked stable.
  - Cation ⟨S²⟩ rises smoothly with temperature for most flagged species; C58H20_3182 jumps (0.95 → 1.15 → 1.27),
    which looks like more than one electronic state.
- **Decision (with the user):** the 9 open-shell pairs are excluded from Phase 1 as whole pairs, in every split:
  `d1/seed_exclusions_open_shell.csv` (6 train, 3 transfer_test). Recorded as an open item: the excluded pairs are
  solo-H-rich (median solo/H 0.42 against 0.08) and large, so the cut acts as a solo-H and size selection (the Wang
  2026 Fig. 1 bias). State it wherever Phase 1 results are reported.
- **Code** (`d2fe2b5` and the commit that adds this note): `dft_seed_run.py` gained `--stab-all --bs`, and `--names`
  now filters before sharding (it used to put all named frames on one process when frame counts aligned with the
  shard count). `d1/transfer_backfill.py` is new (below).

## What arrived

Sent with `tailcat cp` into the user's `tailcat recv` directory. Check the hashes first:

| File | sha256 | Content |
|---|---|---|
| `seed_dft_hangang.tgz` (30.8 MB) | `dd3c845c5811eb5c7a66e864b85c56254618d071e210eb9b4b1ed6be3caef2ea` | `seed_dft/` (7,504 per-frame JSON, 24 process logs), `seed_dft.extxyz` |
| `seed_dft_bs_check.tgz` (0.4 MB) | `8d8091fd782bd1bdab2a97eb58adead54948ccc9da3bfb7993ccae878241bb93` | `bs_check/` (137 broken-symmetry JSON), `bs_check_summary.csv` (24 frames of the 10 flagged pairs), `bs_minima_summary.csv` (all 124 minima), the two name lists |

The originals stay on Hangang under `/appsdata/t641a02/d1/` (not purged). The allocation expires on **2026-11-22**.

## To do on the project node

1. **Unpack** next to the seed inputs:
   ```bash
   S=/var/tmp/$USER-pah/d1/mlff_seed
   sha256sum <inbox>/seed_dft_hangang.tgz <inbox>/seed_dft_bs_check.tgz      # compare with the table
   tar xzf <inbox>/seed_dft_hangang.tgz -C $S                                 # seed_dft/, seed_dft.extxyz
   mkdir -p $S/bs_check_hangang && tar xzf <inbox>/seed_dft_bs_check.tgz -C $S/bs_check_hangang
   ```
   `$S/seed_dft/` may already hold the 202 pilot results; the tarball holds the same 202 files (copied there before
   the run), so overwriting them is harmless.
2. **QA with xTB** (plan, "Quality checks", item 2; the xTB env):
   ```bash
   /var/tmp/$USER-pah/mamba/xtb/bin/python d1/mlff_qa.py $S/seed_dft.extxyz $S/seed_qa --manifest $S/gjf/manifest.json
   ```
   Expect `unstable` on the 50 neutral frames and `s2_review` on the high-⟨S²⟩ cation frames. The new information is
   `energy_outlier` and `dipole_outlier` (state jumps above 0.5 eV), which Hangang could not check (no xTB there).
   Report them per species, and say which fall in excluded pairs.
3. **Apply the exclusions when building the training data.** `seed_dft.extxyz` still holds all 7,504 frames; nothing
   was deleted. A pair is the `source` label without `_q<charge>.xyz`:
   ```python
   import csv
   excl = {r["pair"] for r in csv.DictReader(open("d1/seed_exclusions_open_shell.csv"))}
   keep = lambda info: info["source"].rsplit("_q", 1)[0] not in excl
   ```
   No tool does this yet; build it into the Phase 2 converter (extended XYZ → training format, "Tools still to write").
4. **Phase 1 items still open** (plan, "Quality checks"):
   - G1 force finite-difference check (tool not written). Plan worked out on Hangang: ~10 frames, both charges,
     C10–C60, none excluded, including a high-force boundary frame, a kept high-⟨S²⟩ cation and a Newton-fallback
     frame (C54H30_3837_q1_f012 or C60H22_3811_q1_f025); ~6 components each (3 largest |F| + 3 random), central
     differences at h ≈ 0.005 bohr, displaced SCFs started from the reference density with ⟨S²⟩ checked; target
     ≤ 1e-4 Ha/bohr. The production gradient has no grid response, so expect that to dominate the residual.
   - State-QA side set (item 4). The neutral `mix` variant at the minima is done (`bs_check`). Still open: neutral
     `mix` at the hottest MD frames (use `--stab-all --bs`), cation `huckel`/`mix` guesses, and ROKS forces for the
     three named cations (gpu4pyscf 1.9.2 has ROKS energies but no ROKS gradients; use CPU PySCF).
   - Coverage report (item 5), after the exclusions.

## transfer_test backfill (done on Hangang 2026-11-16)

The exclusion cut `transfer_test` from 8 pairs to 5. The user chose a naive backfill, because the xTB sampler cannot
run on Hangang (no xTB module, no aarch64 `tblite` wheel).

- **Candidates** (`d1/transfer_backfill.py candidates`): 32 unused PAHdb pairs, 8 per 10-C bin (61–100 C), B3LYP/6-31G\*
  in PAHdb for both charges, one per skeleton, no skeleton shared with the seed set, Mai et al. uids excluded; picked
  by farthest-point sampling in edge-type fractions, seeded with the 5 kept transfer pairs. Frames per charge: the
  PAHdb minimum plus 4 Gaussian displacements (σ 0.02/0.03/0.04/0.05 Å), labelled `sampler=naive` and `disp_sigma`.
- **Screen:** the same open-shell criterion as the seed set, on all 32 neutral minima: 7 open-shell (E_RKS − E_BS
  3.6–37.2 mHa, ⟨S²⟩_BS 0.84–2.41; solo-H share 0.33–0.64, again the solo-rich ones), 1 barely unstable
  (C74H22_9921, 0.2 mHa, allowed), 24 stable.
- **Selected** (`transfer_backfill.py select`, 4 most diverse passers per bin): 16 pairs, solo-H share 0–0.60.
  - 61–70: C66H24_3594, C62H20_6671, C62H22_6648, C62H22_6649
  - 71–80: C74H22_6602, C78H30_6585, C72H26_3547, C78H26_3529
  - 81–90: C84H24_174, C90H30_171, C84H28_3381, C84H24_6508
  - 91–100: C92H24_6477, C96H30_3309, C96H28_3373, C100H26_7494
- **DFT:** 160 frames, 4.6 GPU-h; all converged, no Newton fallback. Cation ⟨S²⟩ > 0.80 on 7 frames, at most 0.851
  (C90H30_171_q1_f004), rising with σ. Largest force component 24.3 eV/Å (a σ = 0.05 Å frame).
- **Tarball** `seed_dft_transfer2.tgz` (1.95 MB, sha256
  `083ca247a59fbfc373388bdd845dbc8181d3ee8f638052d42893552321fcc202`): `transfer2/` with `seed_dft/` (160 JSON),
  `seed_dft.extxyz`, `species_transfer2.csv`, `candidates.csv` (all 32 with screen results and the decision),
  `selected_pairs.txt`, `bs_screen/`, `frames_transfer/`, `gjf_transfer/manifest.json`, `xyz/`. The first tailcat
  send failed (the receiver had stopped); resent with `tailcat cp` on 2026-11-16 (sender exit 0, hash not yet
  confirmed on the receiving side). The original stays on Hangang at `/appsdata/t641a02/d1/seed_dft_transfer2.tgz`.
- **On the project node:** run `mlff_qa.py` on `transfer2/seed_dft.extxyz` as for the seed set (manifest
  `transfer2/gjf_transfer/manifest.json`). `transfer_test` is now the 5 kept pairs of `species_v2.csv` plus the 16 of
  `species_transfer2.csv` (21 pairs); `species_v2.csv` itself is unchanged.
- **Limit:** these frames are near-minimum only. Their median RMSD from the minimum is 0.085 Å, against 0.27 Å for
  300 K MD and 0.51 Å for 1200 K MD in the seed set. They add species diversity, not temperature coverage: compare
  transfer errors on `min` and `disp` frames only, and say so where results are reported.
- **Optional:** MD frames for the 16 pairs with the original sampler (`d1/mlff_sample.py`, xTB; one frame at
  300/900/1500 K, as for the original transfer pairs), labelled on Hangang before the allocation ends on 2026-11-22
  (or on the project node's GPUs). That would give real 900–1500 K transfer frames.

## Rules (unchanged)

- Mai et al. geometries (C90H24_618, C144H30_641): never copy or commit them.
- Commit only on `feat-gpu-gh200`; push only when the user asks. Never print `.env`.
- Training data, logs and models stay out of git (size); scripts and plans are committed.
