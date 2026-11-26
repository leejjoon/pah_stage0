# D1 cation temperature rules, version 1 (2026-11-26)

Finite-temperature rules for singly charged PAH cations, built from MLMD on the frozen D1 force model (MACE v8, cation
head) and the polar_v7 dipole model. This follows the coordinator's decoupled 5-GPU protocol (shared notes,
2026-11-26 08:58) and the Step 4 instructions of 16:50.

- **Tables:**
  - `d1/tdep_rules_cation_v1.csv` has the same schema as `tdep_rules_v1_clean.csv`.
  - `d1/perline_widths_cation_v1.csv` has the same schema and family `lo_cm1` values as `perline_widths_v2_clean.csv`.
  - `slice/emission.py --rules-cation d1/tdep_rules_cation_v1.csv --widths-cation d1/perline_widths_cation_v1.csv`
    reads them. Checked with its `load_rules`: it takes all 16 position and strength rules and all 6 width families
    from these tables, and nothing from the neutral tables.
  - Size terms pass its 2-s.e. test only for the 3.3 µm strength and the 11.2 µm position.
- **Neutral species keep `tdep_rules_v1_clean.csv`** (Mai et al. style MLMD). These rules are for cations only.
- **Archive:** `gh200:~/projects/pah-data/cation_mlmd_spectra_v1-2026-11-26.tgz`, described below.

## Data

- **Species:** `d1/cation_mlmd_species_v1.csv`, 100 cations selected by `d1/cation_mlmd_select.py`.
  - All are pure C/H, PAHdb v4 `status == use`, `verdict == ok`, doublets, with Kekulé skeletons and no sp³ carbon,
    and 26 ≤ Nc ≤ 120.
  - 25 per Nc bin (26–40, 41–60, 61–85, 86–120), round-robin over the dominant H-edge class.
- **MD:** `d1/cation_mlmd_produce.py` on gh200 GPU 0, TorchSim with cuEquivariance, MACE v8 (q1).
  - For each species and each of 300, 600 and 900 K: 8 replicas, 16 ps Langevin NVT (γ = 10 ps⁻¹) then 64 ps NVE.
  - MD step 0.5 fs; positions saved every 4 steps (`dt_fs` = 2.0, 32,000 frames).
  - 300 jobs (2,400 replicas) in 9.9 h.
  - Energy drift ≤ 0.06 meV/atom throughout.
  - **2,391/2,400 replicas intact.** The other 9 were each one replica of a job: 3947 at 600 K, and 10734, 10546,
    3947, 9846, 10782, 6454, 3218 and 7452 at 900 K.
    - All 9 have a bond graph at the NVE start that differs from the minimum's, with no blowup.
    - Rule B1 drops them, so each of those jobs keeps 7 replicas.
- **Dipoles:** `d1/cation_mlmd_dipoles.py`, polar_v7 forward-only in float32.
  - Frames are centred at the isotopic centre of mass (H 1.00782503, C 12.0).
  - Computed on wasp33a GPUs 0–3 and gh200 GPU 0, 12:48–22:27.
  - Every `<id>_spec.npz` stores the intact replicas' dipoles [R, 32000, 3].
- **Spectra:** Rule B1: `mlmd_spectra.ir_sigma` maths (qm = 3, unclipped, per-replica NVE temperature), with frequencies
  scaled by 0.9578. Exported by `d1/cation_spec_export.py` into `profile_fit.py`'s format.
  - **Primary (coordinator 16:50):** ACF kept to **L = 16 ps** (8,000 lags at 2 fs). The DCT grid is 0.9985 cm⁻¹
    after scaling, the instrumental resolution of the neutral 200 ps libraries (15,974 lags at 1 fs). No refinement.
  - **Sensitivity case:** L = 5.11 ps, the production `_spec.npz` window (FRACTION 7.987 % of the series; 2,555 lags,
    3.13 cm⁻¹ grid). The spectrum is refined 3× by a cubic spline through the DCT points, so `profile_fit`'s bin-based
    steps behave as for the neutrals.
- **Fits:** `profile_fit.fit_species` per species, run by `d1/cation_rules.py fit`.
  - Regions: oop 650–1000 and in-plane 1100–1700, plus the fixed-window complexes.
  - The truncated-Lorentzian model uses L from each file.

## Method (`d1/cation_rules.py rules`)

These are the neutral table's statistics: per species, the slope against each replica's NVE temperature; then OLS
in log₁₀(Nc/50) over species, giving the value at Nc = 50, the per-decade term, standard errors, the scatter (MAD) and
quad−lin at 1200 K. Per band:

- **3.3 µm:** fixed-window complex (2850–3250), centroid and FWHM, as in the neutral table.
- **6.2 and 7.7 µm, position and strength:** shifting band profiles.
  - The in-plane profile components are tracked from 300 to 900 K (`profile_fit` refits each temperature from the
    previous one).
  - Each component is assigned to a band by its 300 K replica-mean centre in `emission.py`'s FEATURES ranges
    (6.2: 1540–1700; 7.7: 1220–1470; 8.6: 1100–1220).
  - Position is the area-weighted mean centre; strength is the summed area relative to its 300 K value.
  - Width and width300 come from the fixed-window complex rms, as in the neutral table. `emission.py` does not use them.
- **8.6 µm position:** tracked, as for 6.2 and 7.7.
- **8.6 µm strength:** the fixed-window area over 1100–1220. See "8.6 µm strength".
- **11.2, 12.0, 12.7, 13.5 µm:** isolated oop profile components, as in the neutral table. A component that overlaps
  another at 900 K is dropped.

**Tracked-component cut (MAXW = 100 cm⁻¹):** a tracked component whose replica-mean FWHM reaches 100 cm⁻¹ at any
temperature is dropped from its band at every temperature.

- **Mechanism:** a weak, narrow line at 300 K is refitted at 600 and 900 K into a broad pedestal-like component, up to
  the 150 cm⁻¹ bound, that takes up background.
  - Example: C34H16⁺ (uid 7230), 8.6 µm component: area 0.011 → 0.65 and FWHM 4 → 148 cm⁻¹ between 300 and 900 K.
- **Real lines:** median FWHM 5 cm⁻¹ at 300 K, 99th percentile 47–58 cm⁻¹.
- **Dropped components** (of the components assigned to that band; 16 ps / 5.1 ps):
  - 6.2: 3/194 and 4/156
  - 7.7: 50/649 and 37/674
  - 8.6: 8/227 and 10/237
- **Effect:** the cut changes the 6.2 and 7.7 rules little. It removes the four worst 8.6 µm fits. See the second table
  below.
- **Rationale:** the cut is a fit-quality rule of the same kind as the oop overlap drop. Without the cut:
  `cation_rules.py rules --maxw 1e9`.

## Table 1: rules, L = 16 ps (primary) and L = 5.1 ps (sensitivity)

Units:
- position and width: cm⁻¹ per 100 K
- strength: fraction of the 300 K value per 100 K
- width300: cm⁻¹ at 300 K

Columns:
- **at Nc = 50 ± se (n species)**
- **per decade Nc:** the size term, applied by `emission.py` only above 2 s.e.
- **scatter:** between species
- **neutral:** `tdep_rules_v1_clean.csv`, i.e. fixed windows for 6.2, 7.7 and 8.6, and 385 species

| band | quantity | L = 16 ps (primary): at Nc = 50 ± se (n) | per decade Nc | scatter | L = 5.1 ps: at Nc = 50 ± se (n) | 16 − 5.1 | neutral (tdep_rules_v1_clean) |
|---|---|---|---|---|---|---|---|
| 3.3 | position | -6.802 ± 0.111 (100) | +0.392 | 0.876 | -6.792 ± 0.112 (100) | -0.010 | -9.150 |
| 3.3 | width | +2.270 ± 0.130 (100) | +0.592 | 1.057 | +3.801 ± 0.166 (100) | -1.531 | +6.611 |
| 3.3 | strength | +0.018 ± 0.003 (100) | -0.063 | 0.022 | +0.018 ± 0.003 (100) | -0.000 | -0.043 |
| 3.3 | width300 | +20.467 ± 0.478 (100) | -6.755 | 3.424 | +24.321 ± 0.617 (100) | -3.854 | +22.290 |
| 6.2 | position | -4.600 ± 0.166 (95) | -0.145 | 0.792 | -4.764 ± 0.113 (94) | +0.164 | -4.261 |
| 6.2 | width | +2.196 ± 0.053 (95) | -0.910 | 0.374 | +2.216 ± 0.052 (94) | -0.020 | +0.761 |
| 6.2 | strength | -0.050 ± 0.005 (95) | +0.012 | 0.037 | -0.048 ± 0.008 (94) | -0.002 | -0.005 |
| 6.2 | width300 | +19.351 ± 0.308 (95) | +6.069 | 2.684 | +19.251 ± 0.305 (94) | +0.100 | +28.797 |
| 7.7 | position | -2.929 ± 0.323 (100) | +0.271 | 1.894 | -3.380 ± 0.278 (100) | +0.451 | -3.010 |
| 7.7 | width | +0.877 ± 0.083 (100) | -1.100 | 0.843 | +0.877 ± 0.082 (100) | +0.000 | +0.695 |
| 7.7 | strength | -0.006 ± 0.009 (100) | +0.020 | 0.043 | -0.019 ± 0.006 (100) | +0.013 | -0.011 |
| 7.7 | width300 | +61.498 ± 1.001 (100) | +12.937 | 8.709 | +61.422 ± 0.992 (100) | +0.076 | +62.862 |
| 8.6 | position | -2.161 ± 0.173 (94) | +0.420 | 0.884 | -2.398 ± 0.170 (96) | +0.237 | -0.144 |
| 8.6 | width | +0.368 ± 0.062 (94) | +1.153 | 0.545 | +0.377 ± 0.061 (96) | -0.008 | +0.376 |
| 8.6 | strength | +0.020 ± 0.006 (100) | -0.058 | 0.032 | +0.020 ± 0.006 (100) | +0.000 | +0.032 |
| 8.6 | width300 | +27.100 ± 0.415 (94) | -2.540 | 3.837 | +27.198 ± 0.406 (96) | -0.098 | +29.354 |
| 11.2 | position | -1.500 ± 0.156 (67) | -1.462 | 0.422 | -1.449 ± 0.164 (73) | -0.050 | -1.933 |
| 11.2 | width | +0.278 ± 0.303 (67) | -1.160 | 0.568 | +0.457 ± 0.083 (73) | -0.179 | +0.639 |
| 11.2 | strength | -0.047 ± 0.009 (67) | -0.002 | 0.057 | -0.004 ± 0.027 (73) | -0.043 | -0.034 |
| 11.2 | width300 | +5.792 ± 1.725 (67) | -1.047 | 1.200 | +5.621 ± 0.447 (73) | +0.171 | +4.693 |
| 12.0 | position | -1.207 ± 0.165 (51) | -0.848 | 0.285 | -1.266 ± 0.058 (54) | +0.059 | -1.786 |
| 12.0 | width | +0.647 ± 0.119 (51) | -0.419 | 0.567 | +0.709 ± 0.085 (54) | -0.062 | +0.625 |
| 12.0 | strength | -0.021 ± 0.013 (51) | -0.109 | 0.060 | -0.033 ± 0.007 (54) | +0.012 | +0.021 |
| 12.0 | width300 | +4.114 ± 0.256 (51) | -4.074 | 0.908 | +4.972 ± 0.369 (54) | -0.858 | +4.472 |
| 12.7 | position | -0.917 ± 0.116 (58) | +0.352 | 0.391 | -1.156 ± 0.175 (62) | +0.239 | -1.056 |
| 12.7 | width | +0.561 ± 0.129 (58) | -0.187 | 0.573 | +0.545 ± 0.095 (62) | +0.017 | +0.532 |
| 12.7 | strength | -0.031 ± 0.011 (58) | -0.026 | 0.039 | -0.038 ± 0.008 (62) | +0.007 | +0.002 |
| 12.7 | width300 | +4.087 ± 0.483 (58) | -1.790 | 0.911 | +5.503 ± 0.560 (62) | -1.416 | +4.966 |
| 13.5 | position | -1.111 ± 0.118 (60) | +0.372 | 0.277 | -1.099 ± 0.099 (66) | -0.012 | -1.030 |
| 13.5 | width | +0.592 ± 0.081 (60) | -0.610 | 0.474 | +0.678 ± 0.071 (66) | -0.086 | +0.765 |
| 13.5 | strength | -0.012 ± 0.027 (60) | -0.023 | 0.039 | -0.018 ± 0.011 (66) | +0.006 | -0.009 |
| 13.5 | width300 | +3.958 ± 0.238 (60) | -5.541 | 0.939 | +4.207 ± 0.237 (66) | -0.248 | +4.264 |

### L = 16 ps vs 5.1 ps, against the Gate B4 variant sizes

The variant sizes come from `emission.py --b4`: CC strength slope −1.7 and −3.0 % per 100 K; position slopes ±30 %;
width growth ±30 %.

- **Positions:** all within the ±30 % variant.
  - 7.7: 15 % (0.45 cm⁻¹ per 100 K).
  - 8.6: 11 %.
  - 12.7: 21 %.
  - All others ≤ 5 %.
- **CC strengths:** 6.2, 7.7 and 8.6 differ by ≤ 1.3 % per 100 K, inside the 1.7 % variant. The largest is 7.7: −0.6 vs
  −1.9 % per 100 K.
- **oop strengths:** 12.0, 12.7 and 13.5 differ by ≤ 1.2 % per 100 K.
  - **11.2 µm differs by 4.3 % per 100 K** (−4.7 vs −0.4), more than the 3.0 % variant. The two values are 1.5 s.e.
    apart: 0.009 at 16 ps and 0.027 at 5.1 ps.
- **Widths:** slopes and width300 values mostly agree within ±15 %. The exceptions:
  - 12.0 and 12.7 µm width300 are 17 % and 26 % lower at 16 ps (finer grid, narrower fitted lines).
  - **3.3 µm FWHM slope**, about 40 % lower (2.27 vs 3.80), and width300 (20.5 vs 24.3). The finer 16 ps grid resolves the CH-stretch
    complex into narrower structure, so its FWHM is smaller and grows more slowly.
  - **11.2 µm width slope**, about 40 % lower (0.28 ± 0.30 vs 0.46 ± 0.08). At 16 ps this row is poorly determined.
- **Species counts per oop band, 16 ps vs 5.1 ps:**
  - 11.2: 67 vs 73
  - 12.0: 51 vs 54
  - 12.7: 58 vs 62
  - 13.5: 60 vs 66
  - At 1 cm⁻¹ resolution with 8 × 64 ps of data, more oop bands split into close components (noise or real fine
    structure), and these then fail the "isolated at 900 K" test.
  - On the gate species (ovalene⁺), 11.2 µm itself splits into an 880/885 cm⁻¹ pair at 16 ps. Its 1 fs and 2 fs fits
    disagree there, while at 5.1 ps they agree to 3e-5. That per-bin noise is the price of the 16 ps resolution on 64 ps
    runs.
  - The in-plane and 3.3 µm counts are the same at both L.

## 8.6 µm strength

| band | quantity | tracked, unfiltered (n) | tracked, MAXW cut (n) | fixed window (n) | neutral fixed window |
|---|---|---|---|---|---|
| 6.2 | position | -4.653 (95) | **-4.600** (95) | -2.396 (100) | -4.261 |
| 6.2 | strength | -0.050 (95) | **-0.050** (95) | -0.071 (100) | -0.005 |
| 7.7 | position | -3.110 (100) | **-2.929** (100) | -1.603 (100) | -3.010 |
| 7.7 | strength | +0.011 (100) | **-0.006** (100) | -0.035 (100) | -0.011 |
| 8.6 | position | -2.309 (97) | **-2.161** (94) | -0.358 (100) | -0.144 |
| 8.6 | strength | +0.358 (97) | +0.150 (94) | **+0.020** (100) | +0.032 |

Bold marks the values in the primary table. All values are at L = 16 ps.

- **Tracked 8.6 strength is not estimable.**
  - The cation 8.6 µm band is weak: the tracked 300 K area is 0.4–4 % of the total.
  - Area relative to the 300 K area is therefore ill-conditioned whenever the fit moves intensity between
    overlapping weak components as they broaden.
  - Even after the MAXW cut, per-species slopes reach +3.4 per 100 K. The OLS is +0.15 against a median of +0.03, with
    scatter 0.41; the neutral scatter is 0.02.
  - The progression unfiltered → cut → window is +0.36 → +0.15 → +0.020.
- **The window value is consistent with the other estimates.** The fixed-window area (+0.020, scatter 0.03) agrees
  with the tracked per-species median (+0.027) and with the neutral rule (+0.032).
- **It is also the right definition here.**
  - `emission.py` scales every line inside the feature range (1100–1220) by the strength rule, so the window integral
    is the quantity it applies.
  - The 8.6 µm band shifts by only −2.2 cm⁻¹ per 100 K (−13 cm⁻¹ by 900 K), so it stays inside that window.
  - 6.2 µm is different. At −4.6 cm⁻¹ per 100 K its 1562 cm⁻¹ peak crosses 1540 cm⁻¹ by about 900 K. The window then
    loses intensity: fixed −0.071 against tracked −0.050. So 6.2 keeps the tracked strength.
- **Coordinator:** please confirm this splice. Alternatively:
  - `--tracked-strength` restores the tracked row;
  - or drop the row, and `emission.py` falls back to the neutral 8.6 strength.

## Other notes on the table

- **6.2 µm has n = 95 (94 at 5.1 ps).**
  - Four species have no in-plane component with a 300 K centre in 1540–1700: their strongest CC line is at
    1511–1538 cm⁻¹. They are C26H16⁺ (815), C30H18⁺ (4007), C36H18⁺ (3995) and C36H16⁺ (7167).
  - C34H20⁺ (821) has no component near the band.
  - At 5.1 ps, C66H22⁺ (6466) also loses its 6.2 component to the MAXW cut.
- **8.6 µm position has n = 94.**
  - Four species have no component in 1100–1220 at 300 K: 6991, 6641, 6519 and 7434.
  - The MAXW cut removes the only component of two more: 7230 and 9982.
- **6.2 µm position: tracked −4.60, cation fixed window −2.40, neutral fixed window −4.26.**
  - The fixed-window "position" is the argmax of the boxcar-smoothed complex. It jumps between lines within the
    complex, so it under-reports a coherent shift of the whole band.
  - The tracked centre is area-weighted over the band's lines.
  - The same holds for 7.7 µm (−2.93 tracked vs −1.60 window) and 8.6 µm (−2.16 vs −0.36).
- **3.3 µm strength is positive for cations** (+1.8 % per 100 K; neutrals −4.3 %). The cation CH-stretch band is weak
  (6 % of the total at 300 K on the gate species), and its window area grows slightly with T.
- **3.3 µm position, −6.8 cm⁻¹ per 100 K, is smaller than the neutral −9.2.**

## Table 2: per-line widths (`perline_widths_cation_v1.csv`, from L = 16 ps)

Method as `perline_widths_clean.py` for the neutral table, on PAHdb 6-31G* cation sticks and the cation MLMD
(`cation_rules.py widths`).

| family | lo–hi (cm⁻¹) | FWHM 300 K | slope per 100 K | n | L = 5.1 ps: FWHM 300 K, slope (n) | neutral v2 clean |
|---|---|---|---|---|---|---|
| CH stretch (3.3 µm) | 2800–3300 | 10.64 | +1.22 | 100 species | 11.99, +2.48 (100) | 10.57, +5.86 |
| CC stretch (6.2 µm) | 1540–1700 | 6.63 | +2.16 | 15 | 9.22, +1.67 (15) | 14.17, +8.96 (incl. +5.26 extra) |
| 1470–1540 | 1470–1540 | 6.72 | +1.36 | 30 | 8.38, +1.11 (25) | 7.42, +2.03 |
| CC/CH in-plane (7.7 µm) | 1220–1470 | 3.86 | +1.45 | 97 | 4.60, +1.44 (133) | 5.45, +1.65 |
| CH in-plane (8.6 µm) | 1100–1220 | 3.89 | +1.37 | 72 | 4.36, +1.44 (81) | 5.50, +2.38 |
| CH out-of-plane | 650–1000 | 5.79 | +0.28 | 236 | 5.62, +0.46 (255) | 4.69, +0.64 |

- **CH stretch:** obtained by inverting the cation 3.3 µm FWHM rule with `rules_closure`'s bisection. It inherits the
  rule's L dependence: slope 1.22 at 16 ps against 2.48 at 5.1 ps.
- **6.2 µm extra term: not applied.**
  - The neutral method adds an empirical slope k so that broadened sticks reproduce the MLMD growth of the 6.2 µm
    complex rms width.
  - For cations the median gap (rules − MLMD growth, 300 → 900 K) is −10.5 cm⁻¹ at k = 0 and −0.65 at k = 16. It
    crosses zero only at k ≈ 19.4 cm⁻¹ per 100 K (scan to 48, `kscan_ext.py` in the archive).
  - That k would give each 6.2 µm line a FWHM of about 140 cm⁻¹ at 900 K: not line broadening.
  - The cation complex rms growth likely includes the band shifting against the 1540 cm⁻¹ window edge (see "8.6 µm
    strength") and the spread of component shifts.
  - So the table carries the isolated-band slope only, and the scan is capped at k = 16, three times the neutral +5.26.
    This is an uncertainty, like the neutral 7.7 µm note.
- **7.7 µm:** the extra k that would close its growth gap is 5.8 (16 ps) or 6.6 (5.1 ps). It is not applied, as in the
  neutral table.
- **Out-of-plane:** the cation 11.2 µm rule at Nc = 50.

## Resolution and caveats

- **Run length vs ACF window.** 8 × 64 ps NVE per species and T: the ACF window, not the 0.52 cm⁻¹ run-length limit,
  sets the resolution.
  - At L = 16 ps the grid is 0.9985 cm⁻¹, the neutral libraries' resolution, but per-bin noise is higher than in the
    neutral 200 ps runs.
  - At L = 5.1 ps the grid is 3.13 cm⁻¹, and lines narrower than about 3 cm⁻¹ are not resolved.
- **The cation dipole ACF does not decay within the window.**
  - The rotating permanent dipole about the COM keeps it at about 50 % of ACF(0) at 5.1 ps and 11 % at 16 ps.
  - DCT-I of the truncated ACF handles this exactly, which is Rule B1 as written.
  - Zero-padding the truncated ACF (`cation_spec_export.py` before 1e69744) does not: the step at the window edge rings,
    times the ~ω² prefactor, and put 5–90 % of a band's maximum on the bands. No result here uses padded spectra.
- **Profile-fit sensitivity.** Weak 7.7 µm sub-components split differently between fits of spectra that differ by
  1e-4.
  - The tracked aggregation damps this to the per-mille level at rule level on the gate species.
  - The pooled medians agree with the OLS for 6.2 and 7.7 (strength: −0.043 vs −0.050; −0.017 vs −0.006).
- **11.2 µm at 16 ps:**
  - fewer species (67);
  - a width slope with s.e. 0.30;
  - a strength 4.3 % per 100 K away from 5.1 ps.
  - Use with that uncertainty, or take the 5.1 ps row (`rules_L5.csv` in the archive) if the coordinator prefers.

## 1 fs vs 2 fs stride gate (coordinator item 3)

The gate ran on ovalene⁺ (uid 5), 8 replicas × 64 ps at 1 fs. `d1/cation_gate_stride.py` compares the 1 fs series
with every second frame of the same series.

- **dt_fs = 2.0 is honoured end to end.** The 2 fs path reproduces the 1 fs DCT grid exactly; a hard-coded 1.0 would put
  it off by 2×.
- **Spectra:** they agree to ≤ 1.2e-4 of the maximum at 300, 600 and 900 K, at both L.
- **Rule level, L = 5.1 ps:** all bands within 0.2 %.
- **Rule level, L = 16 ps:**
  - 3.3 and 6.2/7.7/8.6 are within 0.7 %, except the 6.2 strength slope: 5 % relative, 0.0016 per 100 K absolute.
  - The oop components split differently (see above).
- **History:**
  - The 13:07 report (0.02–3.4 %, "profile_fit noise floor") used zero-padded 5.1 ps spectra. It was corrected at 17:03
    (shared notes) and rerun here.
  - The coordinator confirmed 2 fs at 16:50.

## Reproduce

```bash
python d1/cation_spec_export.py <stage>/dipoles <out>/spec_L16 --acf-ps 16      # and without --acf-ps for 5.1 ps (--check)
python d1/cation_rules.py fit d1/cation_mlmd_species_v1.csv <out>/spec_L16 <out>/fits_L16.csv --jobs 32
python d1/cation_rules.py rules <out>/fits_L16.csv d1/tdep_rules_cation_v1.csv --fixed <out>/rules_fixed_L16.csv
python d1/cation_rules.py widths <out>/fits_L16.csv d1/tdep_rules_cation_v1.csv d1/cation_mlmd_species_v1.csv \
    <pahdb v4 xml> d1/perline_widths_cation_v1.csv --jobs 32
```

These ran on gh200 CPUs (cores 64–127) with the host python3 (numpy, scipy) and took about 3 min in all.
