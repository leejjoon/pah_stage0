# D1 temperature rules, version 1 clean (2026-11-25)

The same fit as `d1/tdep_rules_v1.md`, on the 385 species of its 549 that are closed-shell singlets with a
spin-consistent PAHdb entry. Use this table for new runs. Requested by the coordinator (2026-11-24).

- **Why:** `d1/screen_spin.py` found that 141 of the 549 species are listed by PAHdb with a multiplicity below the
  minimum their bonding allows (137 of them are the non-Kekule singlets of `pahdb_v4_exclude_nonkekule_v1.csv`), and
  23 more are consistent doublets, outside the closed-shell MLMD model. Their MLMD spectra describe the wrong
  electronic state or a state the model was not trained for.
- **Fit set:** `verdict == ok` and multiplicity 1 in `d1/pahdb_v4_screen_spin_v1.csv`: 385 species (370 C/H, 15 with
  N). Dropped: 141 below the bound, 23 doublets. (The request named 380; the count here is from the screen table.)
- **Input:** the per-species profile fits of version 1, `profile_fits_v1_c{0,1,2}.csv`, copied from Hangang
  (`gh200:~/projects/hangang_appdata/d1/spectra/`). The fit with all 549 species reproduces `tdep_rules_v1.csv`
  exactly. Command: `python d1/profile_fit.py rules d1/tdep_rules_v1_clean.csv <the three tables, filtered>`.
- **Table:** `d1/tdep_rules_v1_clean.csv`, same columns as `tdep_rules_v1.csv`.

Apply as in `tdep_rules_v1.md` ("How to apply"); only table 1 changes.

## Table 1: feature shifts and strengths

Value at Nc = 50; a size term is given where it exceeds 2 standard errors; ± is the scatter between species.

| Feature (µm) | Lines (cm⁻¹) | Position shift (cm⁻¹ per 100 K) | Strength (fraction of its 300 K value per 100 K) | quad−lin at 1200 K: position | Version 1: position | Version 1: strength |
| --- | --- | --- | --- | --- | --- | --- |
| 3.3 | 2800-3300 (CH stretch) | -9.15 (±0.24) | -4.3 -1.2·log₁₀(Nc/50) (±0.3) % | +7.0 | -9.11 (±0.24) | -4.3 -1.1·log₁₀(Nc/50) (±0.3) % |
| 6.2 | 1540-1700 (CC stretch) | -4.26 (±2.23) | -0.5 +2.9·log₁₀(Nc/50) (±1.8) % | +5.1 | -4.01 (±2.44) | -0.4 +2.6·log₁₀(Nc/50) (±1.8) % |
| 7.7 | 1220-1470 (CC/CH in-plane) | -3.01 (±3.54) | -1.1 (±1.0) % | -4.7 | -2.81 (±3.49) | -0.9 -1.3·log₁₀(Nc/50) (±1.0) % |
| 8.6 | 1100-1220 (CH in-plane) | -0.14 -4.41·log₁₀(Nc/50) (±2.35) | +3.2 (±2.1) % | -4.1 | -0.47 -3.43·log₁₀(Nc/50) (±2.12) | +3.1 (±1.8) % |
| 11.2 | 860-925 (CH oop solo) | -1.93 (±0.57) | -3.4 (±8.3) % | -1.5 | -1.90 (±0.61) | -3.8 (±6.8) % |
| 12.0 | 815-860 (CH oop) | -1.79 (±1.28) | +2.1 (±11.3) % | -0.4 | -1.64 (±1.35) | +2.6 (±11.0) % |
| 12.7 | 765-815 (CH oop duo/trio) | -1.06 +1.37·log₁₀(Nc/50) (±0.93) | +0.2 +26.8·log₁₀(Nc/50) (±10.1) % | -2.7 | -0.97 +1.39·log₁₀(Nc/50) (±0.89) | -0.1 +18.7·log₁₀(Nc/50) (±10.3) % |
| 13.5 | 650-765 (CH oop) | -1.03 +2.63·log₁₀(Nc/50) (±0.50) | -0.9 (±6.2) % | -2.7 | -1.03 +2.36·log₁₀(Nc/50) (±0.48) | -1.0 (±7.0) % |

## What changed against version 1

- No value at Nc = 50 moves by more than 0.22 of the scatter between species. In standard errors the largest moves
  are the 7.7 µm strength (−0.9 → −1.1 % per 100 K, 4.2 s.e.), the 3.3 µm width at 300 K (21.5 → 22.3 cm⁻¹, 3.2 s.e.)
  and the 3.3 µm position (−9.11 → −9.15, 3.1 s.e.).
- **Size terms:** the 7.7 µm strength no longer has one (−0.6 ± 0.3 % per decade, below 2 s.e.). The 12.7 µm strength
  term grows from +18.7 to +26.8 % per decade of Nc. The other size terms keep their sign and significance.
- For the 6.2/11.2 ratio the statement of version 1 holds: 11.2 µm loses 3.4 % per 100 K, 6.2 µm 0.5 %.

## Not refitted

- **Per-line widths** (`d1/perline_widths_v2.csv`, table 2 of version 1). The 3.3 µm entry already excludes the
  non-Kekule singlets. The in-plane and out-of-plane entries come from isolated bands of all species
  (`inplane_widths.py`, `profile_fit.py`); they were not redone here, because their inputs are on Hangang.
- **Background** (1700–2800 cm⁻¹) and the validity limits: as version 1. The rules are still for neutral, planar,
  closed-shell PAHs at 300–900 K; applying them to cations is an assumption (Gate B4).

## Use

`slice/emission.py --rules d1/tdep_rules_v1_clean.csv` (the default is still version 1, so that the v1 and v2
prototype runs reproduce). `reports/nc_sr_v3/` is the first run with the clean table.
