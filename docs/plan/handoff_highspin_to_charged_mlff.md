# Handoff: high-spin (non-Kekulé) neutrals into the charged-MLFF work (2026-11-20)

From the D1 session (branch `gh200-containers`) to the charged-MLFF session (branch `feat-gpu-gh200`).

- **The user's decision:** add 10 non-Kekulé neutral PAHs, which have a triplet ground state, and their cations to the
  charged-MLFF seed set.
- **One finding to act on first:** two of your seed-set pairs are non-Kekulé, and their neutral labels are not the
  ground state (section 1).

Rules as in your handoffs:
- commit only on `feat-gpu-gh200`; the files listed here come from `origin/gh200-containers` (section 5);
- never print `.env`;
- DFT results and models stay out of git.

## Background

- **PAHdb's error:** PAHdb v4.00 lists 334 neutral pure C/H PAHs without a Kekulé structure as closed-shell singlets.
  Their intensities come out about 7× too high (`docs/reports/pahdb_v4_nonkekule/README.md` on `gh200-containers`).
- **Why the ground state is high-spin:** these molecules have unequal carbon sublattices. By Lieb's rule, the ground
  state then has spin S = |n_A − n_B|/2, which is a triplet here.
- **What we computed:** the 200 with fewer than 90 C, recomputed in the high-spin state (198 triplets, 2 quintets).
  - Settings: UB3LYP (`b3lypg`)/6-31G* with 6D d functions, (99,590) grid, GPU4PySCF 1.9.2 in container D,
    `CUPY_ACCELERATORS=cub`. These are your seed-set settings, except a tighter `conv_tol` (1e-11).
  - Each geometry was re-optimised, then the analytic Hessian and IR intensities were computed.
  - Checks: no imaginary modes; ⟨S²⟩ at the triplet minima is 2.12–2.28 (median 2.15); 0 failures.
- **Where they are:** `d1/own_spectra/`. These are **our own data, not PAHdb's**:
  - species ids end in `_own` and rows carry `source=own`;
  - the user asked that this marking survive into your datasets (section 4).

## 1. Your pairs C60H20_10225 and C72H22_9911 are triplets

Of the 9 pairs in `d1/seed_exclusions_open_shell.csv`, three are non-Kekulé:
- C60H20_10225 (train);
- C72H22_9911 (transfer_test);
- C96H26_9397 (transfer_test; 90 C or more, so not recomputed).

From transfer2's `candidates.csv`, C96H26_9390 (open-shell, not selected) is non-Kekulé as well. The other 6 excluded
pairs and the other 6 open-shell transfer2 candidates are Kekulé, so they are true diradicaloids.

**The energy comparison:** the energies are on your `f000` minimum, which is the PAHdb geometry. I checked that it is
identical up to a translation, with all interatomic distances equal to 4e-15 Å. Your RKS and broken-symmetry values come
from `os_relabel_1117/dft_seed/<pair>_q0_f000.json`; the triplet values come from our UKS at the same geometry.

| Pair | E(RKS) Ha | E(BS, ⟨S²⟩) Ha | E(triplet, ⟨S²⟩) Ha | Triplet below BS |
| --- | --- | --- | --- | --- |
| C60H20_10225 | −2298.756361 | −2298.775053 (1.047) | −2298.782255 (2.163) | 7.2 mHa (4.5 kcal/mol) |
| C72H22_9911 | −2757.343428 | −2757.363524 (1.042) | −2757.370922 (2.150) | 7.4 mHa (4.6 kcal/mol) |

So the `bs` labels these pairs' neutral frames received in `collect --bs-label` are not the ground state. An Ms = 0
broken-symmetry solution with ⟨S²⟩ ≈ 1 is roughly an equal mix of singlet and triplet.

**Decision for you and the user:** relabel those q0 frames as triplets (mult 3), or take them out of the q0 data.
- Either way, each species should sit on one q0 surface only.
- If C60H20_10225 is one of the 3 formerly excluded pairs behind v2's gain, re-check that gain after the change.
- Frame counts: 32 q0 frames for 10225, 5 for 9911. The cation frames are unaffected.

**Optional check, a few GPU-hours:** a triplet single point on `f000` of the 7 Kekulé open-shell pairs and the 6 Kekulé
open-shell transfer2 candidates. It would confirm that broken-symmetry is the lowest solution there.

## 2. The 10 new species

All files are in `d1/highspin_seed_v1/`, made by `d1/highspin_seed_species.py`.

- **Pool:** the 198 triplets, minus the 2 above, minus 3 with no PAHdb cation. Cations were matched with your
  `graph_hash` on the PAHdb geometries, and the cation formula equals the neutral's for all 10.
- **Selection:** farthest-point sampling in the solo/duo/trio/quartet fractions within each carbon bin, starting from
  C30H14_10770 (the case worked through in the PAHdb report).
- **Skeleton check:** each species' carbon-skeleton hash (`mlff_splits.skeleton`, verified to reproduce your stored
  keys for 10225 and 9911) clashes with no pair in `species_v2.csv` or in transfer2's `candidates.csv`.
- **Splits:** a proposal; re-run your own rules if you prefer.

| Pair | Split | Cation uid (PAHdb state) | Neutral ⟨S²⟩ | Solo / duo / trio / quartet |
| --- | --- | --- | --- | --- |
| C30H14_10770 | train | 10769 (2-A2) | 2.126 | 4/4/6/0 |
| C38H16_10753 | validation | 10767 (2-A") | 2.138 | 4/6/6/0 |
| C38H16_10754 | train | 10752 (2-B1) | 2.143 | 6/4/6/0 |
| C44H20_10622 | train | 10610 (2-A") | 2.139 | 6/8/6/0 |
| C48H18_10537 | train | 10530 (2-A") | 2.126 | 5/10/3/0 |
| C54H20_10440 | train | 3177 (2-AU) | 2.246 | 10/4/6/0 |
| C60H20_10169 | train | 10187 (2-A") | 2.132 | 7/10/3/0 |
| C60H20_10211 | final_test | 10202 (2-A") | 2.141 | 6/14/0/0 |
| C62H20_10176 | transfer_test | 10173 (2-A") | 2.149 | 6/14/0/0 |
| C70H22_9932 | transfer_test | 9959 (2-A") | 2.136 | 8/8/6/0 |

**Files:**
- `species.csv`: `pair, formula, nC, bin, split, uid_q0, mult_q0 (=3), source_q0 (=own), species_own, S2_q0, uid_q1,
  pahdb_state_q1, pahdb_method_q1, n_solo…n_quartet`. Map it to `species_v2.csv`'s columns, adding `skeleton`; keep
  `mult_q0` and `source_q0`.
- `xyz/<pair>_q0.xyz`: **our** triplet minimum, not PAHdb's closed-shell geometry. It is the right minimum for the
  triplet surface.
- `xyz/<pair>_q1.xyz`: the PAHdb cation geometry.

Comment lines follow `mlff_species.py`'s plain style, and all 20 files read with `ase.io.read`.

**Cation spin:** PAHdb lists all 10 cations as doublets. Your AL1 reference frame C60H20_10225_q1_al1_f035 has
⟨S²⟩ 0.850, so a quartet single point at each cation minimum (12 points, under 1 GPU-hour) is worth doing before
sampling.

## 3. Pipeline changes (read in your code, not run)

1. **Sampling:** `mlff_sample.py <pair>_q0.xyz … --charge 0 --mult 3`, with the seed-set settings (32 frames per
   species and charge; 5 for transfer_test).
   - The sampler passes `multiplicity=3` to TBLite, but I have not run GFN2-xTB triplet MD.
   - Check one short run for SCF trouble before the batch.
2. **DFT:** `dft_seed_run.py` already runs UKS for mult > 1 (`spin = mult − 1`).
   - Use `--stab-all` for the triplet frames. Uid 10770 needed 3 Newton fallbacks in 18 SCF calls here, so expect
     more SCF cycles than for doublets.
   - `collect`'s summary counts high ⟨S²⟩ only for doublets (line 149); add triplets.
3. **QA:** `mlff_qa.py` flags `s2_review` when ⟨S²⟩ > 0.80. That is a doublet threshold, and it would quarantine every
   triplet frame.
   - Make it relative to S(S+1): 0.80 = 0.75 + 0.05 for doublets.
   - Triplet minima here reach 2.28, so set the triplet limit from your sampled frames (e.g. 2.0 + 0.35), keeping the
     "group median + 0.05" rule.
   - Groups are formed per species and charge, so separate mult 1 from mult 3 if any relabelled pair keeps both.
4. **Names:** frame names `<pair>_q<q>_f<k>` carry no multiplicity.
   - The 10 new pairs don't collide with anything.
   - Triplet relabels of 10225 and 9911 would reuse existing names. Write them to their own directory and frame names,
     e.g. a `_q0t_` tag; check that `rsplit("_q", 1)` and the manifest still parse.
5. **Marking as our own:** your extxyz `source` key already holds the xyz file name. Add a separate key to every frame
   of these species, e.g. `ref_origin=own_highspin_v1` (minimum from our UKS triplet, not PAHdb), and keep it through
   `collect`, QA and dataset building.

## 4. Model

- **Why a separate head:** the MACE heads have no spin input, so a triplet neutral cannot share the q0 head (the
  closed-shell surface).
- **Suggestion:** a separate head for the triplet neutral, e.g. `q0t`, on the shared backbone. The data will be small:
  10 species × 32 frames, plus the relabels. Alternatives that take spin directly: AIMNet2-NSE or MACE-OMOL (OMol25
  `spin` = multiplicity), which you have already timed.
- **Dipoles:** needed as for the other states.
- **Validation reference (G5):** `d1/own_spectra/nonkekule_highspin_v1_sticks.csv`, not PAHdb. PAHdb's entries for
  these species are the wrong state.
  - Point `mlff_accept.py symfreq` at it for these species.
  - Frequencies are given unscaled, and scaled with PAHdb v4.00's own per-mode factors (0.9794 below 1111 cm⁻¹, 0.9691
    for 1111–1800, 0.9597 above).
  - The other 188 triplets are an unseen test set for size transfer.

## 5. Where it runs, and cost

**Where each step runs:**
1. Sampling: the project node (xTB environment).
2. DFT: Hangang, container D, `slurm_dft_seed.sbatch`, **after AL1 and without taking AL1's slots**, before the
   allocation ends on 2026-11-22. The project node's GPUs work if Hangang is too tight.
3. QA and training: the project node.

**Frames and cost:**
- New species: 8 pairs × 2 charges × 32 + 2 transfer_test pairs × 2 × 5 = 532.
- Relabels: 37 frames.
- Basis for the cost: our warm-started triplet energy+force points took a median of 112, 172, 236 and 339 s at
  30–45, 45–60, 60–75 and 75–90 C. With cold starts and stability tests, assume about 1.5× that.
- Total: ~35–50 GPU-hours, about 5–7 h on one node at 2 processes per GPU.

**Getting the files** (nothing to send by tailcat):

```bash
git fetch origin
git checkout origin/gh200-containers -- d1/highspin_seed_v1 d1/own_spectra d1/highspin_seed_species.py \
    docs/plan/handoff_highspin_to_charged_mlff.md
```

On Hangang the same files are also in `/appsdata/t641a02/pah_stage0` (branch `gh200-containers`). The triplet JSONs
(Hessian-level results with geometries) are in `/appsdata/t641a02/d1/triplet_v1/<uid>.json`. Don't merge the branches.

## 6. Results: the DFT run on Hangang (2026-11-21, done by the D1 session)

**How it ran:** sampling and labelling ran on Hangang in one job of `d1/slurm_highspin_seed.sbatch`
(`gh200-containers`), job 1316386, 5.7 h on one node, after test job 1316132.
- Your code was used unchanged from a detached worktree of `feat-gpu-gh200` at `c93e4fc`: `mlff_sample.py`,
  `g16_io.py write`, then `dft_seed_run.py run --stab-all` and `collect`.
- xTB came from a conda env with tblite 0.8.0 and ASE 3.29. Check these against the project node's versions.

**What was produced:**
- **528 frames:** 8 pairs × 2 charges × 32, plus the 2 transfer pairs × 2 × 4 (min + MD at 300/900/1,500 K, as for
  transfer2's MD).
- **Provenance:** every frame has `ref_origin=own_highspin_v1` (q0, mult 3, from our triplet minimum) or
  `ref_origin=pahdb_v4.00` (q1, mult 2).

**Checks:**
- **Convergence:** all converged, none needed the Newton fallback, all 528 were stability-tested.
  - q0: 264 stable.
  - q1: 262 stable, 2 re-optimised (C44H20_10622_q1_f013, C48H18_10537_q1_f026).
- **⟨S²⟩, triplets:** 2.119–2.475, median 2.151.
- **⟨S²⟩, cations:** 0.797–1.082, median 0.816.
- **C54H20_10440 is the outlier** in both charges: q0 up to 2.475, q1 up to 1.08. Its triplet minimum already sits at
  2.246. Review it before training.
- **Bond graph:** 33 frames changed it.
- **Force scale:** largest |F| component has a median of 5.75 eV/Å (max 22.3, `disp`), against 5.57 in your seed set.
- **GPU time:** 31.5 GPU-process-h in all.
- **Energy check:** C30H14_10770 q0 `f000` equals our own triplet E_opt to 1e-6 Ha.

**QA warning:** `collect`'s doublet check (> 0.80) flags 250 of the 264 cation frames. Cations of these species sit
at 0.80–0.82 even at the minimum (C30H14_10770 q1 `f000`: 0.801). So the `mlff_qa.py` 0.80 rule would quarantine
nearly all of them, as well as every triplet frame (section 3, item 3). Use per-group (median + 0.05) or per-mult
limits.

**Not done:**
- `mlff_qa.py` (it needs your xTB setup to compare against);
- the quartet check of the cation minima;
- the relabels of C60H20_10225 and C72H22_9911.

**Package:** `/appsdata/t641a02/d1/highspin_seed_run.tgz` (3.2 MB), sha256
`ca1de823e248b3b0d0b5a5bb6504f38844f3d5cbaf7e4ab88d1878461591385e`. It holds `highspin_seed_run/` with:
- `frames/`: 20 extxyz;
- `gjf/manifest.json`;
- `dft/`: 528 JSONs;
- `highspin_seed_dft.extxyz`: the `collect` output;
- `logs/`.

The user sends it to the project node.
