# Neutral + cation symbolic regression, version 5: production run with the cation MLMD rules (2026-11-27)

Requested by the coordinator (shared notes, 2026-11-27 00:09). Follows `reports/nc_sr_v4/README.md`. Spectra and
measurements ran on gh200 (container C, CPU only, 48 niced workers, 29 minutes); the regression ran on z2mini
(38 minutes). Driver: `slice/run_nc_v5.sh`. Orion continuum checks: `slice/orion_cont.py`. All tables of this page,
with the v4 formulas and the size and photon-energy estimates of the templates: `tables.md` (`slice/report_v5.py`).

## Summary

| Question | Answer |
| --- | --- |
| What changed? | Charged species now take the cation MLMD tables (`d1/tdep_rules_cation_v1.csv`, `d1/perline_widths_cation_v1.csv`, L = 16 ps). Neutrals keep the clean tables. In v4 both charges used the neutral tables |
| Do the results change? | Hardly. Forest test errors of the four main sets move by at most 0.003 in f, 0.002 dex in size and 0.13 eV in photon energy. The formulas keep their form (same ratios, constants within a few per cent) |
| Best results (forest, local integration with 3.3 µm) | f 0.049, log Nc 0.042 dex, E 0.74 eV (v4: 0.049, 0.044, 0.78). Formulas of complexity ≤ 12: 0.056, 0.058 dex, 1.21 eV |
| Does the 11.2 µm slope matter (−4.7 against −0.4 % per 100 K)? | No. Applying the models to spectra made with the other slope shifts f by −0.002 to −0.007, size by ≤ 0.002 dex and photon energy by −0.05 to −0.11 eV. Retraining on those spectra gives the same test errors (within 0.001, 0.001 dex, 0.03 eV) |
| Freer PAHFIT continuum under 6–9 µm | PAHFIT does not use it. Five hot blackbodies (400–1,500 K) and two broad plateau profiles end at zero or negligible amplitude in the four PDR templates and at a few per cent in the H II template. Band ratios change by < 0.5 % (7 % in H II), and f by 0.00 (0.03 in H II). PAHFIT's high bands are not caused by a stiff continuum |
| Which method carries the bias? | Not decided by fit quality. The two differ in how they assign the emission between the bands: PAHFIT gives it to Drude wings, local integration to the continuum, and the library has no such emission. Local integration is the stabler estimate: f moves by ≤ 0.03 when its blue anchor is moved, while PAHFIT's f falls from 0.61–0.75 to 0.27–0.34 when the local continuum is forced on it. I recommend the monotone local value, with PAHFIT's as an upper bound (section 4) |
| Curved continuum for 17 µm | Works on the templates: a parabola in log–log through the four anchors follows the continuum to 3–4 %, and the 17 µm complex is 0.13–0.16 of the 11.2 µm band in the four PDR templates, inside the library range (−0.05 to 0.16), though at its upper edge. The H II template gives −0.12 (outside). The ratio adds nothing to the regression (section 5) |
| Orion Bar templates, f | Local, MIRI: 0.36 (H II), 0.55–0.64 (PDR). Local with 3.3 µm: 0.34, 0.43–0.59. PAHFIT: 0.61–0.63, 0.70–0.75. All within 0.03 of v4. No template is outside the library in any ratio used |

## 1. Run

| Stage | What | Code |
| --- | --- | --- |
| Emission | 2,386 species, 7 photon energies. Neutrals: `tdep_rules_v1_clean.csv` + `perline_widths_v2_clean.csv`. Charged: `tdep_rules_cation_v1.csv` + `perline_widths_cation_v1.csv`; every position, strength and width row comes from the cation tables. Second variant `s112` with the cation 11.2 µm strength slope −0.004 | `emission.py --rules-cation --widths-cation --alt s112:11.2:-0.004` |
| Populations | the 2,000 populations of v3 and v4 (same labels) | `populations.py --design v3` |
| Measurement | PAHFIT and local integrals for both variants; harmonic and noise-free controls; Orion templates; 30 Monte Carlo tables; endpoint control. New: `Lcrv_17` | `measure.py` |
| Regression | forest, single ratio and PySR for the monotone local sets (`Hn`, `H33n`) and the PAHFIT sets (`Pn`, `P33n`); forest only for the other sets; a second run trained on the `s112` table | `regress_v2.py` |
| v4 comparison | PySR for `Pn` and `P33n` was not run in v4. It was run now on the v4 tables (`results_v4_pfit.json`) | |

Not repeated: the plateau and wing tables of v4.

How much the cation tables move the spectra: in populations with f > 0.7, the local 6.2, 7.7 and 8.6 µm ratios to
11.2 µm rise by 8–10 % against v4, 12.7 µm falls by 5 % and 3.3 µm rises by 21 %. Below f = 0.3 the changes are
≤ 3 %. These are small against the spread of the ratios over the library (a factor of 10–20).

## 2. Test errors and formulas

Held-out 300 populations. "Lpch": band integrals above the monotone local continuum. "NIRSpec + MIRI": with the
3.3 µm band. Left value v4 (cations with neutral rules), right value v5.

| Observables | Predictor | f: v4 | v5 | log Nc (dex): v4 | v5 | E (eV): v4 | v5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lpch, MIRI | forest | 0.065 | 0.067 | 0.047 | 0.045 | 1.49 | 1.62 |
| Lpch, MIRI | best single ratio | 0.097 | 0.093 | 0.077 | 0.078 | 2.42 | 2.41 |
| Lpch, MIRI | formula, complexity ≤ 7 | 0.100 | 0.096 | 0.067 | 0.067 | 1.84 | 1.92 |
| Lpch, MIRI | formula, complexity ≤ 12 | 0.073 | 0.072 | 0.056 | 0.058 | 1.33 | 1.40 |
| Lpch, NIRSpec + MIRI | forest | 0.049 | 0.049 | 0.044 | 0.042 | 0.78 | 0.74 |
| Lpch, NIRSpec + MIRI | best single ratio | 0.097 | 0.093 | 0.077 | 0.078 | 1.43 | 1.30 |
| Lpch, NIRSpec + MIRI | formula, complexity ≤ 7 | 0.096 | 0.093 | 0.067 | 0.067 | 1.33 | 1.32 |
| Lpch, NIRSpec + MIRI | formula, complexity ≤ 12 | 0.067 | 0.056 | 0.055 | 0.058 | 1.06 | 1.21 |
| PAHFIT, MIRI | forest | 0.083 | 0.082 | 0.042 | 0.042 | 2.55 | 2.43 |
| PAHFIT, MIRI | best single ratio | 0.096 | 0.093 | 0.062 | 0.062 | 2.60 | 2.59 |
| PAHFIT, MIRI | formula, complexity ≤ 7 | 0.098 | 0.096 | 0.058 | 0.056 | 2.62 | 2.57 |
| PAHFIT, MIRI | formula, complexity ≤ 12 | 0.088 | 0.084 | 0.047 | 0.049 | 2.54 | 2.53 |
| PAHFIT, NIRSpec + MIRI | forest | 0.072 | 0.069 | 0.041 | 0.041 | 0.67 | 0.70 |
| PAHFIT, NIRSpec + MIRI | best single ratio | 0.096 | 0.093 | 0.062 | 0.062 | 1.31 | 1.26 |
| PAHFIT, NIRSpec + MIRI | formula, complexity ≤ 7 | 0.098 | 0.102 | 0.058 | 0.056 | 1.30 | 1.27 |
| PAHFIT, NIRSpec + MIRI | formula, complexity ≤ 12 | 0.073 | 0.080 | 0.052 | 0.049 | 0.95 | 0.94* |
| – | constant (no information) | 0.304 | 0.304 | 0.108 | 0.108 | 2.66 | 2.66 |

\* One of the 300 test populations has a negative PAHFIT 12.7 µm ratio, for which this formula (it contains
`log(r127)`) is undefined; the error is over the other 299.

Formulas of v5 (ratios to the 11.2 µm band: `h…` local, `r…` PAHFIT; `h33`, `r33` the 3.3 µm band). All
deterministic runs reproduced; no run crashed. The last column is the best validation error with shuffled labels
(no better than a constant).

| Observables | Target | Formula (complexity) | Test | Forest | Reproduced; shuffled-label val |
| --- | --- | --- | ---: | ---: | --- |
| Lpch, MIRI | f | `square(1.1835997 - (sqrt(h62) / h77))` (7) | 0.096 | 0.067 | yes; 0.276 |
| Lpch, MIRI | f | `square(1.263863 - (sqrt(h62) / h77)) - (h127 * 0.2735684)` (11) | 0.072 | 0.067 | yes; 0.276 |
| Lpch, MIRI | log Nc (dex) | `((h127 / 2.1022692) - h135) + 1.7228187` (7) | 0.067 | 0.045 | yes; 0.109 |
| Lpch, MIRI | log Nc (dex) | `(1.0196404 + sqrt(h127 + (h86 / (h77 / 1.6858175)))) - h135` (12) | 0.058 | 0.045 | yes; 0.109 |
| Lpch, MIRI | E (eV) | `(h62 + -0.1998465) / (h77 / 18.46752)` (7) | 1.92 | 1.62 | yes; 2.54 |
| Lpch, MIRI | E (eV) | `square((((h127 - -4.381673) * h62) + -0.7974225) / (h77 - h86))` (12) | 1.40 | 1.62 | yes; 2.54 |
| Lpch, NIRSpec + MIRI | f | `square(h86) / (square(h86) + 0.3170931)` (7) | 0.093 | 0.049 | yes; 0.276 |
| Lpch, NIRSpec + MIRI | f | `square(square((sqrt((h33 + h127) * 0.21926527) / h77) + -1.0343404))` (12) | 0.056 | 0.049 | yes; 0.276 |
| Lpch, NIRSpec + MIRI | log Nc (dex) | `1.7218648 - (h135 + (h127 * -0.4772367))` (7) | 0.067 | 0.042 | yes; 0.107 |
| Lpch, NIRSpec + MIRI | log Nc (dex) | `(sqrt((h86 / (h77 / 1.6718836)) + h127) - h135) - -1.0219655` (12) | 0.058 | 0.042 | yes; 0.107 |
| Lpch, NIRSpec + MIRI | E (eV) | `square(square(h33 + -1.2001607) + -3.309451)` (7) | 1.32 | 0.74 | yes; 2.52 |
| Lpch, NIRSpec + MIRI | E (eV) | `((h127 / h120) + 8.884191) * sqrt(h33 + (h86 * 0.07009197))` (12) | 1.21 | 0.74 | yes; 2.52 |
| PAHFIT, MIRI | f | `(-3.599821 / (r77 + 2.467603)) + 1.2449372` (7) | 0.096 | 0.082 | yes; 0.271 |
| PAHFIT, MIRI | f | `square(((-0.94823104 - (r127 / 2.2212386)) / (r77 + 0.75741005)) + 1.0857233)` (12) | 0.084 | 0.082 | yes; 0.271 |
| PAHFIT, MIRI | log Nc (dex) | `sqrt(r127 - -2.9183004) - square(r135)` (7) | 0.056 | 0.042 | yes; 0.111 |
| PAHFIT, MIRI | log Nc (dex) | `((sqrt(r127) * 0.43657055) + 1.5848602) - square(sqrt(r135) + -0.16762118)` (12) | 0.049 | 0.042 | yes; 0.111 |
| PAHFIT, MIRI | E (eV) | `((r62 * -3.7914743) + r77) + 7.2388477` (7) | 2.57 | 2.43 | yes; 2.53 |
| PAHFIT, MIRI | E (eV) | `r77 + (log(r77) + ((1.6544539 - (r62 + r135)) * 5.002635))` (12) | 2.53 | 2.43 | yes; 2.53 |
| PAHFIT, NIRSpec + MIRI | f | `sqrt(log(r77 + 0.9561428)) - 0.59573907` (7) | 0.102 | 0.069 | yes; 0.274 |
| PAHFIT, NIRSpec + MIRI | f | `(sqrt(r77) - 0.52583563) - ((r127 + (r33 + r77)) * 0.15139449)` (12) | 0.080 | 0.069 | yes; 0.274 |
| PAHFIT, NIRSpec + MIRI | log Nc (dex) | `sqrt(r127 + 2.918326) - square(r135)` (7) | 0.056 | 0.041 | yes; 0.107 |
| PAHFIT, NIRSpec + MIRI | log Nc (dex) | `2.1866071 - (square(r135 * 1.2401565) - (-0.29673493 / (r127 + 0.49359992)))` (12) | 0.049 | 0.041 | yes; 0.107 |
| PAHFIT, NIRSpec + MIRI | E (eV) | `((r33 - r135) * 7.1761603) + 5.3613086` (7) | 1.27 | 0.70 | yes; 2.46 |
| PAHFIT, NIRSpec + MIRI | E (eV) | `log(r127) + square(square(r135 - (r33 + -1.0454509)) + -3.5269709)` (12) | 0.94 (299 of 300; undefined for one) | 0.70 | yes; 2.46 |

Against v4 (`tables.md`): the same leading ratios in every case (7.7 µm for f, 12.7 and 13.5 µm for size, 3.3 µm
for photon energy). Two formulas changed form at equal error: local MIRI f (now `sqrt(h62)/h77` inside the square)
and PAHFIT MIRI photon energy, which stays uninformative (2.5 eV against 2.66 for a constant).

Monte Carlo scatter at S/N 50 (300 test populations, 10 noise draws):

| Observables | Predictor | f | log Nc (dex) | E (eV) |
| --- | --- | ---: | ---: | ---: |
| Lpch, MIRI | forest | 0.027 | 0.009 | 0.50 |
| Lpch, MIRI | formula ≤ 12 | 0.026 | 0.009 | 0.42 |
| Lpch, NIRSpec + MIRI | forest | 0.023 | 0.008 | 0.23 |
| Lpch, NIRSpec + MIRI | formula ≤ 12 | 0.022 | 0.009 | 0.32 |
| PAHFIT, MIRI | forest | 0.018 | 0.007 | 0.27 |
| PAHFIT, MIRI | formula ≤ 12 | 0.018 | 0.006 | 0.11 |
| PAHFIT, NIRSpec + MIRI | forest | 0.018 | 0.007 | 0.22 |
| PAHFIT, NIRSpec + MIRI | formula ≤ 12 | 0.021 | 0.006 | 0.20 |

Endpoint control (plan 5.9, control 2) with the cation tables: unchanged from v4. Without noise, local integration
recovers f to 0.003–0.008 at Nc 30, 50 and 80; PAHFIT to 0.014–0.18, worst at Nc 30 (`endpoint.json`).

## 3. Sensitivity to the cation 11.2 µm strength slope

The cation rules give −4.7 % per 100 K at L = 16 ps and −0.4 at 5.1 ps (`d1/tdep_rules_cation_v1.md`). The `s112`
spectra use −0.4 for the 11.2 µm lines of the charged species only. For f > 0.7 this lowers every ratio to 11.2 µm
by 5 %.

Held-out 600 populations. 'applied' = the model trained on the 16 ps spectra applied to the 5.1 ps spectra; 'retrained' = trained and tested on the 5.1 ps spectra.

| Observables | Predictor | f: 16 ps | applied | shift | retrained (test) | log Nc (dex): 16 ps | applied | shift | retrained (test) | E (eV): 16 ps | applied | shift | retrained (test) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lpch, MIRI | forest | 0.068 | 0.068 | -0.006 | 0.068 | 0.046 | 0.046 | +0.000 | 0.045 | 1.64 | 1.62 | -0.07 | 1.62 |
| Lpch, MIRI | formula ≤ 12 | 0.071 | 0.071 | -0.005 | – | 0.055 | 0.055 | -0.002 | – | 1.45 | 1.46 | -0.08 | – |
| Lpch, NIRSpec + MIRI | forest | 0.050 | 0.051 | -0.007 | 0.049 | 0.042 | 0.042 | +0.000 | 0.043 | 0.77 | 0.79 | -0.10 | 0.74 |
| Lpch, NIRSpec + MIRI | formula ≤ 12 | 0.055 | 0.056 | -0.004 | – | 0.055 | 0.055 | -0.002 | – | 1.11 | 1.12 | -0.11 | – |
| PAHFIT, MIRI | forest | 0.082 | 0.082 | -0.007 | 0.081 | 0.041 | 0.041 | +0.000 | 0.042 | 2.38 | 2.36 | -0.06 | 2.46 |
| PAHFIT, MIRI | formula ≤ 12 | 0.086 | 0.086 | -0.005 | – | 0.048 | 0.048 | -0.001 | – | 2.46 | 2.46 | +0.02 | – |
| PAHFIT, NIRSpec + MIRI | forest | 0.071 | 0.073 | -0.006 | 0.070 | 0.038 | 0.039 | +0.000 | 0.041 | 0.68 | 0.68 | -0.05 | 0.68 |
| PAHFIT, NIRSpec + MIRI | formula ≤ 12 | 0.082 | 0.082 | -0.002 | – | 0.048 | 0.048 | -0.001 | – | inf | inf | +nan | – |

"Shift" is the mean change of the prediction. The largest effects are −0.007 in f, 0.002 dex in size and −0.11 eV:
a tenth of the test error or less. The choice between the two slopes does not need to be settled for Stage 0.

## 4. Orion check 1: PAHFIT with a freer continuum under 6–9 µm

`slice/orion_cont.py` refits the five templates (and 10 noise draws each) with four packs: classic; `hot` (classic
plus modified blackbodies at 400, 500, 700, 1,000 and 1,500 K); `plat` (classic plus two broad Drude profiles at 6.6
and 7.8 µm, FWHM 1.5 µm, counted as continuum); `hotplat` (both). Figure: `figs/fig_orion_cont.png`.

| | H II | APDR | DF1 | DF2 | DF3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reduced χ², classic | 52.6 | 68.5 | 62.5 | 62.5 | 63.6 |
| Reduced χ², hot / plat / hotplat | 51.3 / 52.7 / 51.1 | 68.5 / 69.7 / 68.5 | 62.5 / 63.0 / 62.5 | 62.3 / 62.5 / 62.3 | 63.6 / 63.6 / 63.6 |
| Largest change of 6.2, 7.7, 8.6 / 11.2 (hotplat) | 7 % | 0 % | 0 % | 0.5 % | 0.2 % |
| Continuum share of the flux at 7.7 µm: PAHFIT (any pack) | 0.31 | 0.09 | 0.09 | 0.13 | 0.17 |
| The same, monotone local continuum | 0.44 | 0.21 | 0.21 | 0.29 | 0.36 |
| 7.7/11.2: PAHFIT powers | 2.78 | 3.52 | 4.19 | 4.10 | 4.59 |
| 7.7/11.2: local integral of PAHFIT's band model | 2.30 | 3.02 | 3.50 | 3.26 | 3.50 |
| 7.7/11.2: local integral of the template | 1.76 | 2.35 | 2.71 | 2.38 | 2.39 |
| Share of the anchor flux that PAHFIT assigns to band profiles, 5.5 / 5.9 / 9.7 µm | 0.45 / 0.51 / 0.05 | 0.82 / 0.78 / 0.16 | 0.88 / 0.75 / 0.17 | 0.87 / 0.66 / 0.13 | 0.84 / 0.59 / 0.11 |
| Reduced χ², 5.3–10.2 µm: classic → local continuum forced | 74 → 165 | 100 → 202 | 85 → 173 | 81 → 132 | 93 → 116 |
| f (forest, PAHFIT MIRI): classic / hotplat / local continuum forced | 0.61 / 0.62 / 0.28 | 0.71 / 0.71 / 0.29 | 0.72 / 0.72 / 0.34 | 0.71 / 0.71 / 0.30 | 0.75 / 0.75 / 0.27 |
| f (forest, local MIRI): standard anchors / blue anchor at 5.03–5.12 µm | 0.36 / 0.24 | 0.60 / 0.59 | 0.64 / 0.64 | 0.58 / 0.58 | 0.55 / 0.55 |

What this shows:

- **PAHFIT's continuum is not the limit.** The extra components start at non-zero amplitude and the fit sets them to
  zero. The continuum under the 6–9 µm bands stays a convex blackbody curve (the 200 K component), half as high as the
  local one.
- **The gap is the emission between the bands.** PAHFIT fills 6.6–7.0, 8.2 and 9–10 µm with Drude wings and the wide
  7.7 µm component, and takes 59–88 % of the flux at the two blue anchors (5.5 and 5.9 µm) as band emission in the PDR
  templates. Local integration calls all of that continuum. Of the factor 1.5–1.9 between the two 7.7/11.2 ratios, a
  factor 1.2–1.3 is lost wings (rows 6 and 7) and the rest is the continuum shape (rows 7 and 8).
- **Forcing the local continuum on PAHFIT** makes the fit worse (χ² in 5.3–10.2 µm up by a factor 1.2–2.2), and the
  solution is not usable: the bands become several times the flux, with strong attenuation. No fit here is adequate
  in absolute terms (χ² 50–70 for 1 % errors), so this is weak evidence against the local continuum, within a model
  that assumes Drude profiles.
- **The library cannot decide.** Its lines are narrow (FWHM ≤ 30 cm⁻¹), so its spectra have almost no emission between
  the bands, and both methods recover the true band ratios there (medians 0.76–0.97 for PAHFIT and 0.84–0.95 for the
  local integrals, relative to the noise-free windows). Whether the inter-band emission of the templates belongs to
  the same carriers is outside what Stage 0 can test.

**Verdict.** The requested test answers one half: a freer continuum does not lower PAHFIT's estimate, so PAHFIT's
f = 0.61–0.75 is not a continuum-stiffness artefact. It does not show that the local value is biased either. My
recommendation is to quote the monotone local value (f = 0.43–0.64 for the PDR templates, 0.34–0.36 for H II) and
PAHFIT's as the upper bound, for two reasons. The local integrals measure the same thing in the library and in the
templates (band cores), whereas PAHFIT's bands include inter-band emission that the training spectra lack. And the
local estimate is stable: moving the blue anchor to 5.03–5.12 µm changes f by ≤ 0.03 in the PDR templates (0.12–0.15 in
H II, where the 5 µm continuum is different), while PAHFIT's f spans 0.27–0.75 depending on the continuum. The
systematic uncertainty of f on real spectra stays about ±0.15.

Caveat on the blue-anchor test: on library spectra the moved anchor changes the 6.2/11.2 ratio by −12 % (median), so
the v5 model is not strictly calibrated for it. It is a sensitivity test, not a second estimate.

## 5. Orion check 2: curved local continuum for 17 µm

`measure.py` now also writes `Lcrv_17`: the 16.0–17.9 µm integral above a parabola in (log wavelength, log flux)
fitted to the four anchor medians at 14.6–15.4 and 18.5–20.0 µm. Figure: `figs/fig_17um_continuum.png`. In v4 the
monotone continuum gave −0.35 to 0.06 for the templates and PAHFIT 0.62 to 1.40, both outside the library.

| 17 µm / 11.2 µm | H II | APDR | DF1 | DF2 | DF3 | library 1–99 % |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Curved continuum | −0.12 | 0.14 | 0.16 | 0.13 | 0.13 | −0.05 to 0.16 |

- The parabola misses the anchors by 3–4 %, with the same sign pattern in every template, so the true curvature is
  not exactly quadratic. With a continuum of 1–3 times the 11.2 µm peak, that is a systematic error of roughly ±0.05 in
  the ratio.
- The four PDR templates are now inside the library range, at its upper edge (DF1, 0.159, just below the 99 % point, 0.164). The H II
  template is not.
- The ratio carries little: adding it to the monotone sets (`Hc`, `Hc33`) leaves the forest errors unchanged or
  slightly worse (f 0.069 against 0.067; E 1.78 against 1.62 eV without 3.3 µm), and the template f moves by ≤ 0.03.
  The production sets stay without 17 µm.

## 6. Orion Bar templates

Cation fraction: forest ± Monte Carlo scatter | formula of complexity ≤ 12. Reported, never a gate. Size and photon
energy are in `tables.md`.

**f**

| Observables | Version | HII | APDR | DF1 | DF2 | DF3 | outside the library |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Lpch, MIRI | v4 | 0.38 ± 0.01 \| 0.37 | 0.61 ± 0.00 \| 0.58 | 0.64 ± 0.00 \| 0.64 | 0.61 ± 0.01 \| 0.57 | 0.58 ± 0.00 \| 0.56 | none |
| Lpch, MIRI | v5 | 0.36 ± 0.01 \| 0.42 | 0.60 ± 0.00 \| 0.60 | 0.64 ± 0.00 \| 0.64 | 0.58 ± 0.02 \| 0.60 | 0.55 ± 0.02 \| 0.59 | none |
| Lpch, NIRSpec + MIRI | v4 | 0.37 ± 0.01 \| 0.39 | 0.57 ± 0.00 \| 0.54 | 0.61 ± 0.00 \| 0.56 | 0.51 ± 0.00 \| 0.49 | 0.44 ± 0.00 \| 0.48 | none |
| Lpch, NIRSpec + MIRI | v5 | 0.34 ± 0.01 \| 0.39 | 0.55 ± 0.00 \| 0.55 | 0.59 ± 0.00 \| 0.58 | 0.48 ± 0.01 \| 0.51 | 0.43 ± 0.01 \| 0.49 | none |
| PAHFIT, MIRI | v4 | 0.59 ± 0.01 \| 0.57 | 0.69 ± 0.00 \| 0.66 | 0.73 ± 0.00 \| 0.70 | 0.73 ± 0.00 \| 0.70 | 0.75 ± 0.00 \| 0.74 | none |
| PAHFIT, MIRI | v5 | 0.61 ± 0.00 \| 0.56 | 0.71 ± 0.00 \| 0.65 | 0.72 ± 0.00 \| 0.69 | 0.71 ± 0.00 \| 0.69 | 0.75 ± 0.00 \| 0.73 | none |
| PAHFIT, NIRSpec + MIRI | v4 | 0.64 ± 0.00 \| 0.61 | 0.70 ± 0.00 \| 0.69 | 0.74 ± 0.00 \| 0.73 | 0.73 ± 0.00 \| 0.72 | 0.75 ± 0.00 \| 0.74 | none |
| PAHFIT, NIRSpec + MIRI | v5 | 0.63 ± 0.00 \| 0.60 | 0.70 ± 0.00 \| 0.69 | 0.72 ± 0.00 \| 0.74 | 0.72 ± 0.00 \| 0.72 | 0.75 ± 0.00 \| 0.76 | none |

Compare the emission-weighted cation share with Ricca et al. 2025, Table 3, before quoting any of these.

## 7. What is left for later stages

1. The inter-band emission of real spectra (section 4). A library with anharmonic profiles, or training spectra
   built from observed band profiles, would let the two methods be compared on equal terms.
2. The 6.2 µm line-width term of the cations (no closing value; `d1/tdep_rules_cation_v1.md`) and the 7.7 µm one of
   both charges are carried as uncertainties.
3. Photon energy without 3.3 µm stays poor (1.4–2.5 eV against 2.66 for a constant).

## Files

| File | Content |
| --- | --- |
| `tables.md` | every table, v4 next to v5, with all formulas |
| `results_base.json`, `results_s112.json` | `regress_v2.py` output: trained on the 16 ps spectra, and on the `s112` spectra |
| `results_v4_pfit.json` | PySR for the PAHFIT sets on the v4 tables |
| `features_base.csv`, `fronts/` | features per population; Pareto fronts of the deterministic PySR runs |
| `endpoint.json` | endpoint control |
| `orion_cont.json`, `orion_cont.log` | Orion refits |
| `figs/` | `fig_orion_cont.png`, `fig_17um_continuum.png` (`slice/figures_v5.py`) |
| `provenance.txt` | hosts, container and table hashes, versions |

Raw outputs: `gh200:/var/tmp/jjlee-pah/sr_proto/out_v5/`, `z2mini:/var/tmp/jjlee-pah/sr_v5/`; archived as
`sr_nc_proto_v5-2026-11-27.tgz` (and `sr_nc_proto_v5_h5-2026-11-27.tgz` for the HDF5 spectra) in
`gh200:~/projects/pah-data/`.
