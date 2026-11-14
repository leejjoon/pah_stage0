# IR intensity convention for MD spectra, and how they enter the emission cascade (2026-11-12)

Response to committee review finding B1 (`docs/plan/charged_mlff_committee_review.md`). Decision: MD spectra use the
harmonic quantum correction (`qm=3`), no clipping of negative values, and each replica's own mean NVE temperature in
the prefactor. They are kept in absolute units and enter the cascade as absorption cross-sections net of stimulated
emission, multiplied by the Planck function once. Code: `d1/mlmd_spectra.py` (production default),
`d1/tdep_compare.py`; checks: `d1/estimator_check.py`, `d1/estimator_xtb_check.py`.

## The estimator

`mlmd_spectra.ir_sigma` (the authors' `calc_MD_IR` maths) computes, from the classical dipole autocorrelation C(t),

σ(ω) = (1 / 6ħcε₀) · ω (1 − e^(−x)) · Ĉ(ω) · Q(x),  x = ħω / kT,

with Ĉ the cosine transform of the first 7.987 % of C(t) and Q an optional quantum correction: qm=0 none, 1 is
2/(1 + e^(−x)), 2 is x/(2 tanh(x/2)), 3 is the harmonic correction x/(1 − e^(−x)). Negative values are set
to 0, and frequencies are scaled by 0.9578.

For a harmonic mode the classical autocorrelation is proportional to kT/ω², so only Q = x/(1 − e^(−x)) gives a
temperature-independent integrated intensity, as quantum mechanics requires for a harmonic oscillator (absorption net
of stimulated emission). With qm=0 each band is weighted by (1 − e^(−x))/x, which tends to kT/ħω at low T: bands are
weighted roughly by 1/ν, and the weight of high-frequency bands grows with T.

## Validation

**Classical harmonic oscillators** (`estimator_check.py`; four modes at 500, 893, 1613, 3060 cm⁻¹ scaled, known
dipole derivatives, mode energy kT, 100 ps at 1 fs). Ratio of the estimated to the exact harmonic intensity
A = 42.2561 |dμ/dQ|² km/mol, unclipped:

| T (K) | qm=0, 893 / 1613 / 3060 cm⁻¹ | qm=3, 893 / 1613 / 3060 cm⁻¹ |
| --- | --- | --- |
| 50 | 0.037 / 0.020 / 0.011 | 0.984 / 0.985 / 0.990 |
| 300 | 0.218 / 0.122 / 0.065 | 0.985 / 0.985 / 0.990 |
| 1200 | 0.593 / 0.423 / 0.253 | 0.984 / 0.985 / 0.990 |

The I(1613)/I(893) ratio, relative to 300 K, is 1.000 at every T with qm=3 and grows to 1.088 / 1.190 / 1.274 at
600 / 900 / 1200 K with qm=0, exactly the analytic (1 − e^(−x))/x weight. Clipping the negative ripples that the
truncated autocorrelation puts around sharp lines inflates band integrals by up to 2× (2.05 at 893 cm⁻¹, 300 K); it
matters most for narrow lines, i.e. at low T, and so adds a spurious temperature trend of its own. The residual 1–2 %
deficit with qm=3 is the ±40 cm⁻¹ integration window.

**A real molecule, same potential on both sides** (`estimator_xtb_check.py`): GFN2-xTB harmonic intensities at the xTB
minimum against band integrals of the GFN2-xTB MD spectra at 300 K (4 replicas, 100 ps NVE; `xtb_test`):

| Species | Total MD / harmonic, qm=0 | Total MD / harmonic, qm=3 | Isolated bands, qm=3 |
| --- | --- | --- | --- |
| naphthalene | 0.142 | 1.167 | oop 773 cm⁻¹ 1.01 |
| naphthalene⁺ | 0.242 | 1.141 | 0.66–1.21 for 357–1442 cm⁻¹; CH stretch 0.38–0.47 |
| coronene | 0.128 | 1.330 | oop 826 cm⁻¹ 0.81 |
| coronene⁺ | 0.150 | 0.930 | oop 795 cm⁻¹ 0.97; CC 1341–1480 0.58–0.65; CH stretch 0.28–0.33 |

With qm=0 the ratios follow (1 − e^(−x))/x (CH stretch 0.07–0.09, out-of-plane 0.2–0.3 at 300 K). With qm=3 the
isolated bands come out within ~±20 % of harmonic; the remaining differences are anharmonic effects in the xTB MD
itself (electrical anharmonicity, intensity moved out of the window), not estimator error. One of them is physical and
relevant: in xTB the cation CH-stretch band carries only ~30–45 % of its harmonic intensity already at 300 K.

## Decision

- **qm=3, unclipped, replica NVE mean temperature in the prefactor**, for every MD spectrum used for intensities. The
  authors' convention is kept only to reproduce their published files (`mlmd_spectra.py --check`).
- Centroids and widths are computed from the non-negative part of the spectrum (shape); integrated intensities and
  shares from the full, unclipped spectrum (the ripples cancel over a band window).
- Band positions and widths do not depend on the choice: the slopes in the xTB/MLMD comparison change by ≤ 0.3 cm⁻¹
  per 100 K between conventions.

## Absolute normalisation

σ(ν̃) is an absorption cross-section in cm² per molecule. Its integral over a band in true (unscaled) wavenumber,
∫σ dν̃ (cm per molecule) × N_A × 10⁻⁵, is the band intensity in km/mol, directly comparable with PAHdb's harmonic
intensities. Spectra stay in these units for the cascade; peak-normalised spectra (as published, and as the
`mean` field of `mlmd_spectra.py` output) lose the absolute strength and must not be used for band ratios across
molecules or temperatures. The 0.9578 frequency scaling moves band positions only; integrals are taken over the
unscaled axis.

## How σ(ν, T) enters the emission cascade

- The qm=3 MD σ(ν, T) is the absorption cross-section **net of stimulated emission** at internal temperature T. In the
  harmonic limit its band integrals are the temperature-independent harmonic intensities; what it adds is the
  anharmonic change of positions, widths and band strengths with T.
- In the thermal (Kirchhoff) approximation the emitted power per unit frequency at internal temperature T is
  σ(ν, T) · B_ν(T). The thermal population of the emitting levels enters **once, through the Planck function** (or its
  microcanonical counterpart in the cascade). It must not enter a second time through the cross-section: with qm=0
  spectra it would, because their (1 − e^(−x))/x weighting is itself a thermal factor.
- In the cascade, at each step's internal temperature T(E), σ is interpolated between the computed temperatures
  (300/600/900/1200 K, band by band in position, width and integrated strength); below 300 K the 300 K or 50 K
  spectrum is used, above 1200 K extrapolation is flagged.
- The canonical MD temperature stands in for the microcanonical temperature of the cascade, as in the PAHdb standard
  emission model; the error of that substitution is largest for the smallest species and is part of the D1
  systematics, not of this convention.
- The PAHdb harmonic cascade uses integrated harmonic intensities (T-independent) with an empirical 15 cm⁻¹ redshift
  and fixed profile (Bauschlicher et al. 2018, §4); the MD σ(ν, T) replaces the profile, the shift and the
  temperature-dependent part of the strength, with the same Planck factor.

## Consequences for results so far

- **Mai et al.'s published spectra** (`IR_txt_*K_qm0`) use qm=0: relative band intensities within a spectrum are
  weighted roughly by 1/ν (at 300 K the CH stretch at ~0.07–0.09 and the out-of-plane bands at ~0.2–0.3 of harmonic),
  and ratios of high- to low-frequency bands rise with T by the factor above. Their band positions are unaffected.
  Our step 2b comparison with their naphthalene spectrum used the same convention on both sides, so it stands as a
  comparison of positions and of relative shares within one convention.
- **xTB charge test** (`charged_tdep_validation.md` §3.3; `xtb_test/compare_new.txt` vs `compare_old.txt`, now also
  with the integrity filter: 5 trajectories left out). Share slopes per 100 K, 300–900 K, old → new convention:

  | Share | Old (qm=0, clipped, nominal T) | New (qm=3) |
  | --- | --- | --- |
  | Neutral CH stretch, xTB | +1 to +1.3 % | −1.4 to −2.4 % |
  | Cation CH stretch, xTB (coronene / naphthalene / pyrene) | +3 / +13 / +18 % | +0.7 / +7 / +24 % |
  | Neutral in-plane, xTB vs MLMD (control) | +6–10 % vs 0–4 % | +11–16 % vs 0.2–2.8 % |
  | Cation in-plane, xTB | −0.9 to −1.6 % | −2.2 to −3.1 % |

  The convention changes intensity trends by up to 2× and can flip their sign. The cation CH-stretch rise survives,
  but xTB fails the MLMD control in the in-plane region by more than before, so xTB intensity trends remain
  untrustworthy; the conclusion on positions and widths is unchanged.
- **Neutral MLMD production** (`cluster_migration.md`): spectra are now computed with this convention by default.

## Still open

- Compare the MLMD model's own harmonic intensities (model Hessian and EPNN dipole derivatives) with its low-T MD
  spectra, and both with PAHdb, once the charged-state models exist (gate G5).
- The cascade implementation itself (D1, T0b slice) must take σ in absolute units and apply the Planck factor once;
  this document is the specification it is checked against.
