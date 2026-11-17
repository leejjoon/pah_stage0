# D1 temperature rules, version 1 (2026-11-18)

How PAH band positions, line widths and strengths change with temperature, for D1's emission model: apply to PAHdb /
surrogate harmonic lines at the temperature of each step of the cooling cascade. Derived from the Hangang MLMD
libraries (549 planar neutral PAHs with spectra at 300, 600 and 900 K; Mai et al. 2025's model; production spectral
convention). Sources: `d1/tdep_rules_v1.csv` (bands), `d1/perline_widths_v2.csv` (line widths); analysis and caveats in
`docs/plan/d1_hangang_new_species.md` (sections "Band trends" to "Empirical width term").

## How to apply

At temperature T (K), with dT = (T − 300)/100 and Nc the number of carbon atoms:
1. **Shift** every harmonic line by (shift of its family) × dT. A line takes the family of the feature whose range it
   falls in; outside all ranges, no shift.
2. **Broaden** every line with a Lorentzian of FWHM = FWHM₃₀₀ + growth × dT of its line family (table 2).
3. **Scale** the intensities of each feature by 1 + (strength change) × dT (table 1); keep the harmonic intensities at
   300 K.
4. **Background:** add a broad component in 1700-2800 cm⁻¹ (overtones and combinations, absent from harmonic lines):
   about +0.9 % of the total intensity per 100 K above 300 K (≈ 1.6 % at 300 K, 6.9 % at 900 K).

## Table 1: feature shifts and strengths

Value at Nc = 50; a size term is given where it exceeds 2 standard errors; ± is the scatter between species.

| Feature (µm) | Lines (cm⁻¹) | Position shift (cm⁻¹ per 100 K) | Strength (fraction of its 300 K value per 100 K) | quad−lin at 1200 K: position |
| --- | --- | --- | --- | --- |
| 3.3 | 2800-3300 (CH stretch) | -9.11 (±0.24) | -4.3 -1.1·log₁₀(Nc/50) (±0.3) % | +7.1 |
| 6.2 | 1540-1700 (CC stretch) | -4.01 (±2.44) | -0.4 +2.6·log₁₀(Nc/50) (±1.8) % | +4.9 |
| 7.7 | 1220-1470 (CC/CH in-plane) | -2.81 (±3.49) | -0.9 -1.3·log₁₀(Nc/50) (±1.0) % | -4.0 |
| 8.6 | 1100-1220 (CH in-plane) | -0.47 -3.43·log₁₀(Nc/50) (±2.12) | +3.1 (±1.8) % | -4.1 |
| 11.2 | 860-925 (CH oop solo) | -1.90 (±0.61) | -3.8 (±6.8) % | -1.4 |
| 12.0 | 815-860 (CH oop) | -1.64 (±1.35) | +2.6 (±11.0) % | -0.5 |
| 12.7 | 765-815 (CH oop duo/trio) | -0.97 +1.39·log₁₀(Nc/50) (±0.89) | -0.1 +18.7·log₁₀(Nc/50) (±10.3) % | -2.9 |
| 13.5 | 650-765 (CH oop) | -1.03 +2.36·log₁₀(Nc/50) (±0.48) | -1.0 (±7.0) % | -2.7 |

## Table 2: per-line widths (Lorentzian FWHM)

| Line family | Range (cm⁻¹) | FWHM at 300 K (cm⁻¹) | FWHM growth (cm⁻¹ per 100 K) | Basis |
| --- | --- | --- | --- | --- |
| CH stretch (3.3 µm) | 2800-3300 | 10.6 | 5.9 | fitted so broadened PAHdb lines reproduce the MD 3.3 µm feature width (60 species) |
| CC stretch (6.2 µm) | 1540-1700 | 14.3 | 9.3 | isolated MD bands (+3.7) plus an empirical +5.6 that closes the 6.2 µm width growth |
| 1470-1540 cm⁻¹ | 1470-1540 | 7.4 | 2.1 | isolated MD bands (37) |
| CC/CH in-plane (7.7 µm) | 1220-1470 | 5.9 | 1.75 | isolated MD bands (145) |
| CH in-plane (8.6 µm) | 1100-1220 | 6.0 | 2.5 | isolated MD bands (90) |
| CH out-of-plane (11.2-13.5 µm) | 650-1000 | 4.5 | 0.67 | isolated MD bands (~1,800) |

## Uncertainties and validity

- **Validity:** neutral, planar PAHs, Nc 22-216, 300-900 K. Not covered: non-planar and superhydrogenated species (the
  model breaks down; see the screens), charged species, and the 334 PAHdb v4.00 non-Kekule species (excluded).
- **Charged species:** positions and widths transfer from neutrals within a few cm⁻¹ (xTB test,
  `charged_tdep_validation.md`); strengths are not established.
- **Extrapolation to 1200 K** (linear): add the last column of table 1 as an uncertainty on positions (1-3 cm⁻¹ for oop
  bands, 4-7 for 3.3-8.6 µm). Widths: up to ~8 cm⁻¹ for the 3.3 µm feature.
- **7.7 µm complex:** the rules under-predict its width growth by ~3.6 cm⁻¹ (rms) between 300 and 900 K (~6 % of its
  width); not closable with a width term.
- **6.2 µm line width:** the growth of 9.3 cm⁻¹ per 100 K is partly empirical (+5.6 fitted); it reproduces the population
  mean, with a per-species scatter of about ±4 cm⁻¹.
- **Model:** all numbers come from one ML force field and dipole model (Mai et al. 2025), classical dynamics.
- **For the 6.2/11.2 ratio:** in these spectra 11.2 µm loses ~4 % of its strength per 100 K while 6.2 µm stays flat, so
  6.2/11.2 rises ~4-5 % per 100 K at fixed charge.
