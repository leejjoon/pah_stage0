# Diagnostic-impact gate B4: prototype result (2026-11-12)

Response to committee review finding B4 (`docs/plan/charged_mlff_committee_review.md`): does the charge dependence of
band-strength temperature trends change what a 6.2/11.2 µm ratio says about the ionised fraction, by more than a
tolerance fixed in advance? Code: `d1/b4_impact.py`; outputs `/var/tmp/$USER-pah/d1/b4/run{1,2,3}*`.

**Answer, provisional:** with charge-dependent trends of the size GFN2-xTB suggests, yes: the recovered cation
fraction is biased by up to 0.03–0.14 (broad band windows), above the 0.05 tolerance for species up to 60 C at
photon energies ≥ 8 eV. With half that charge dependence the bias stays within tolerance (≤ 0.037). The deciding
quantity is the temperature slope of the cation CC-band strength, which xTB measures poorly; the pilot's first job is
to measure it at DFT level (the naphthalene-cation DFT MD now running gives the first value).

## Set-up

- **Species:** all 1,181 PAHdb v4.00 pairs of a closed-shell neutral and a doublet cation of the same structure with
  20–100 C (harmonic B3LYP/6-31G\* spectra, all 3N−6 modes).
- **Emission:** thermal cascade. A photon of E₀ = 6, 8, 10 or 12 eV heats the molecule to T_max (harmonic heat
  capacity from all modes); while it cools by dE = C(T) dT, band i receives the fraction σᵢ(T)·B(νᵢ, T) / Σⱼ σⱼ(T)·B(νⱼ, T)
  (absorption net of stimulated emission times the Planck function, counted once, as in
  `docs/plan/intensity_convention.md`). Checked against the PAHdb suite's own `cascade()` at 8 eV for coronene and
  circumcoronene, neutral and cation: per-mode energy fractions agree within 2.4×10⁻⁴, band-window fractions to 3–4
  decimals.
- **Variants:**

  | Variant | Band strengths | Positions and widths |
  | --- | --- | --- |
  | v1 (calibration) | harmonic, fixed | −15 cm⁻¹, FWHM 15 cm⁻¹ (PAHdb standard model) |
  | v2 | × (1 + s (T − 300 K)/100 K) with the MLMD neutral slopes, both charges | shift with T (CC −5, oop −2.5, CH −10.5 cm⁻¹ per 100 K), FWHM +1 cm⁻¹ per 100 K |
  | v3 | as v2, cations with the xTB cation slopes | as v2 |
  | v2s, v3s | as v2, v3 | as v1 (strength effects only) |
  | ×0.5, ×2 | cation−neutral slope difference halved or doubled | |

  Slopes (fraction per 100 K, absolute band strength, harmonic-corrected MD spectra, 300–900 K): MLMD neutral
  (naphthalene, pyrene, coronene mean) out-of-plane +0.8 %, in-plane +2.2 %, CH stretch −1.9 %; xTB cation mean
  out-of-plane +2.9 %, in-plane −0.9 %, CH stretch +15 %. Factors are held at their 1200 K value above 1200 K and
  clipped to [0.5, 2].
- **Diagnostic:** band energy per absorbed photon in two window conventions (narrow: 1570–1650 and 870–910 cm⁻¹;
  broad: 1500–1700 and 850–940 cm⁻¹), summed over the species of a size bin. The cation fraction f (of absorbed
  photons) is recovered by inverting R(f) = 6.2/11.2 calibrated with v1, applied to R from the other variants;
  bias = f_recovered − f_true for f_true = 0.1–0.9.
- **Tolerance fixed in advance:** |bias| ≤ 0.05 in f.

## Results

Max |bias| over f_true = 0.1–0.9, broad windows:

| Size (C) | E₀ (eV) | v2s | v3s | v3s×0.5 | v3s×2 | v2 (with shifts) | v3s − v2s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20–29 | 6 / 8 / 10 / 12 | 0.019–0.029 | 0.065–0.106 | 0.021–0.037 | 0.168–0.268 | 0.067–0.255 | 0.084–0.136 |
| 30–39 | 6 / 8 / 10 / 12 | 0.020–0.026 | 0.044–0.084 | 0.017–0.028 | 0.114–0.218 | 0.022–0.055 | 0.059–0.109 |
| 40–59 | 6 / 8 / 10 / 12 | 0.017–0.026 | 0.032–0.067 | 0.014–0.021 | 0.082–0.174 | 0.014–0.020 | 0.044–0.089 |
| 60–100 | 6 / 8 / 10 / 12 | 0.015–0.028 | 0.023–0.050 | 0.013–0.024 | 0.059–0.129 | 0.037–0.039 | 0.032–0.069 |

(Ranges run over E₀ from 6 to 12 eV; the strength effects grow with E₀, v2 does so only for ≤ 39 C. Per-bin
numbers: `run2.txt`.)

- **Neutral strength trends applied to both charges (v2s)** bias f by ≤ 0.03: within tolerance.
- **The charge-dependent part (v3s − v2s)** adds 0.03–0.14, above tolerance for ≤ 60 C at ≥ 8 eV and for 20–39 C at
  all E₀; within tolerance for 60–100 C at 6–8 eV. It scales roughly linearly with the assumed cation−neutral slope
  difference.
- **Band shifts** dominate v2 for the smallest, hottest species (20–29 C: up to 0.26) and saturate the narrow
  windows (|bias| = 0.9 for most bins, i.e. the inversion hits f = 0 or 1): at T_max of 1500–2000 K a −5 cm⁻¹ per 100 K
  CC shift moves the 6.2 µm band out of an 80 cm⁻¹ window. That is a statement about window-based measurement and
  position modelling (positions transfer between charges, §3.3 of the proposal), not about charge dependence;
  PAHFIT-type fits with free band centres would behave differently and are the next test.
- **Which band class drives the charge effect** (signed bias change against v2s at f = 0.5, `run3`): in-plane CC
  bands about half (−0.013 to −0.032), out-of-plane about 30 % (−0.008 to −0.018), CH stretch about 20 %
  (−0.003 to −0.013); the parts add up to the total within 0.003. So the result rests mainly on the cation CC-band
  strength slope (xTB −0.9 % per 100 K against +2.2 % for neutral CC bands in MLMD), not on xTB's least reliable
  number (the cation CH stretch, +15 % per 100 K on a band that is weak at 300 K).

## Reading against the gate

- The committee's question is whether the correction materially improves the inference. On this prototype it does if
  the true cation-minus-neutral difference in CC-band strength slopes is ≳ 2 % per 100 K (the xTB size, ~3 %, gives
  0.03–0.14), and it does not if it is ≲ 1.5 % per 100 K (half the xTB size gives ≤ 0.037).
- So the gate cannot be decided from xTB. It is decided by a DFT-level number: the CC-band strength slope of a
  cation, from (a) the naphthalene-cation DFT MD running now (B3, part one), then (b) the pilot models on the seed
  set. A pilot result is therefore worth having before large-scale expansion, as the review staged it.

## Limits of this prototype

- Thermal (canonical) cascade approximation, as in the PAHdb standard model; no microcanonical correction.
- One band-ratio diagnostic (6.2/11.2), two fixed-window conventions; no PAHFIT decomposition, no continuum, noise
  or instrument model; populations are single size bins with one photon energy.
- Band classes get one slope each, from three species (and xTB for cations); no size dependence of the slopes.
- f is the fraction of absorbed photons, not of molecules (UV cross-sections of neutrals and cations differ).
- Slopes above 900 K are extrapolated (to 1200 K, held beyond).
