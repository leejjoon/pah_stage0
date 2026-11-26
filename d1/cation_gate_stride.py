# 1 fs vs 2 fs dipole-stride gate of the cation MLMD rules (coordinator 2026-11-26 08:58, item 3), on identical
# dynamics: the gate species' 1 fs dipoles (cation_mlmd_dipoles.py work --gate -> dipoles_gate/<uid>_T<T>_fine_spec.npz)
# and the same series taken every second frame (exactly the 2 fs production frames). Both go through the production
# chain: Rule B1 spectra (ir_sigma, qm=3, unclipped, per-replica T_nve; ACF of the production length, 2555 lags at
# 2 fs; the 1 fs ACF gets 5109 lags so both span 5108 fs on one DCT grid: with int(F x FRACTION) lags each, 5111 vs
# 2555, the grids differ by 0.02 % and the fit split weak components differently, a gate artefact), cation_spec_export (3x padded grid), profile_fit.fit_species (complexes 3.3/6.2/7.7/8.6, oop and in-plane
# profile components), and per-band T-slopes as profile_fit.per_species. Gate: the oop and in-plane profile components
# (shift, width, strength slopes per 100 K within 1 % of the 1 fs slope, with floors 0.02 cm^-1, 0.01 cm^-1 and 0.03 (the strength rule tolerance); 300 K
# values within 1 %, positions within 0.01 cm^-1). The fixed-window complexes are printed, not gated (the coordinator
# asked for shifting profiles, not fixed windows).
# Reported: (1) the spectra (max pointwise difference of the replica-mean B1 spectra, limit 1 %); (2) the rule-level band
# quantities (cation_rules.per_species: tracked 6.2/7.7/8.6 area-weighted centre and summed area, 3.3 complex, isolated
# oop components); (3) the individual profile components (limit 1 %). On the ovalene+ gate (2026-11-26) (1) is 9e-5,
# (2) slopes within 3.4 % (300 K values within 0.3 %; the 3.3 um centre within 0.11 cm^-1) and (3) fails on weak 7.7/12.0 sub-components: profile_fit's decomposition responds to a 1e-4 random
# perturbation of the same spectra by up to 2 % at rule level, so (2)-(3) measure fit sensitivity, not the stride.
# --acf-ps: the ACF window, 16 ps by default (the Step 4 primary; no refinement) or 5.11 (production _spec.npz; spectra
# spline-refined 3x). Until 4bbdd38 the 5.1 ps spectra were zero-padded, which distorts them (cation_spec_export.py header).
#   python cation_gate_stride.py <dipoles_gate dir> <work dir> [--uid 5] [--acf-ps 16|5.11]
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlmd_spectra as ms  # noqa: E402
import cation_spec_export as ex  # noqa: E402
import profile_fit as pf  # noqa: E402
import cation_rules as cr  # noqa: E402

TS = (300, 600, 900)


def tree(Z, root, stride, fraction=None):
    """Spectra of the series taken every stride-th frame. The ACF lag count is matched across strides: the 2 fs
    series has n2 = int(F2 x fraction) lags (as in production), the 1 fs one 2 (n2 - 1) + 1, so both span the same
    time and the DCT-I grids (spacing 1 / (2 dt (n - 1))) coincide exactly; L is the 2 fs value in both."""
    frac = fraction or ex.FRACTION
    for T, z in Z.items():
        sp = str(z["species"]); dt = float(z["dt_fs"]) * stride; F = z["dipole"].shape[1] // stride
        F2 = z["dipole"].shape[1] * float(z["dt_fs"]) // 2; n2 = int(F2 * frac)
        n = int(round((n2 - 1) * 2 / dt)) + 1; L = n2 * 2.0 * 1e-15
        old = ex.FRACTION; ex.FRACTION = (n + 0.5) / F; out, raw = [], []
        try:
            for mu, t in zip(z["dipole"], z["T_nve"]):
                out.append(ex.ir_sigma_padded(mu[stride - 1::stride], float(t), dt, 15974.0))
                raw.append(ex.ir_sigma_padded(mu[stride - 1::stride], float(t), dt, 1.0))   # k = 1: unpadded
        finally:
            ex.FRACTION = old
        os.makedirs(f"{root}/{T}", exist_ok=True)
        np.savez(f"{root}/{T}/{sp}_{T}K.npz", freq=out[0][0], sigma=np.array([s for _, s in out]), prefactor_T=z["T_nve"],
                 replicas=z["replicas"], acf_len_s=L)
        yield T, raw


def b1_windows(spec):
    f = spec[0][0]; s = np.mean([x[1] for x in spec], 0); df = f[1] - f[0]; tot = s.sum() * df
    return {b: s[(f >= lo) & (f < hi)].sum() * df / tot for b, lo, hi in pf.BANDS} | {"total": tot, "_f": f, "_s": s}


def slopes(rows):
    """per (region, band or comp): {quantity: slope per 100 K} from the replica rows, as profile_fit.per_species."""
    by = {}
    for r in rows:
        if r["replica"] != "mean":
            by.setdefault((r["region"], r["band"] if r["region"] == "complex" else r["comp"]), []).append(r)
    res = {}
    for k, v in by.items():
        b300 = [r for r in v if r["T"] == 300]
        if len(b300) < 3 or not b300[0]["band"]:
            continue
        x = np.array([float(r["T_nve"]) for r in v]) / 100; cx = k[0] == "complex" and b300[0]["band"] != "3.3"
        qs = {"position": "peak" if cx else "center", "width": "rmsw" if cx else "fwhm", "strength": "win" if k[0] == "complex" else "area"}
        out = {}
        for name, q in qs.items():
            y = np.array([float(r[q]) for r in v]); y300 = np.mean([float(r[q]) for r in b300])
            out[name] = (np.polyfit(x, y / y300 if name == "strength" else y, 1)[0], y300)
        res[(k[0], b300[0]["band"], k[1])] = out
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("gate"); ap.add_argument("work"); ap.add_argument("--uid", default="5")
    ap.add_argument("--acf-ps", type=float, default=16.0, help="ACF window: 16 (primary, 8,000 lags at 2 fs) or 5.11 (production _spec.npz)")
    a = ap.parse_args()
    Z = {T: dict(np.load(f"{a.gate}/{a.uid}_T{T}_fine_spec.npz")) for T in TS}
    sp = str(Z[300]["species"])
    print(f"{sp}: {[Z[T]['dipole'].shape for T in TS]} at {float(Z[300]['dt_fs'])} fs; T_nve " +
          ", ".join(f"{T}: {np.mean(Z[T]['T_nve']):.0f}" for T in TS))
    F2 = Z[300]["dipole"].shape[1] * float(Z[300]["dt_fs"]) / 2; frac = (round(a.acf_ps * 500) + 0.5) / F2
    print(f"ACF: {int(F2 * frac)} lags at 2 fs ({int(F2 * frac) * 2e-3:.2f} ps), 1 fs {2 * (int(F2 * frac) - 1) + 1} lags")
    runs = {"1fs": (1, frac), "2fs": (2, frac)}
    fits, b1 = {}, {}
    for name, (stride, frac) in runs.items():
        root = f"{a.work}/{name}"
        b1[name] = {T: b1_windows(s) for T, s in tree(Z, root, stride, frac)}
        rows = pf.fit_species(sp, {T: f"{root}/{T}" for T in TS}, [("oop", 650, 1000), ("inplane", 1100, 1700)], complexes=True)
        with open(f"{a.work}/{name}_fits.csv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        fits[name] = slopes(rows)
    worst = 0.0; lines = []
    g1, g2 = (float(np.diff(b1[k][300]["_f"][:2])[0]) for k in ("1fs", "2fs"))
    print(f"B1 fixed-window fractions (unpadded, replica mean), 2 fs / 1 fs - 1 (grids {g1:.6f} / {g2:.6f} cm^-1; information only):")
    for T in TS:
        d = {b: b1["2fs"][T][b] / b1["1fs"][T][b] - 1 for b in b1["1fs"][T] if b[0] != "_" and b1["1fs"][T][b] > 1e-3 * (b != "total")}
        print(f"  {T} K: " + ", ".join(f"{b} {v:+.1e}" for b, v in d.items()))
    FLOOR = {"position": 0.02, "width": 0.01, "strength": 0.03}      # per 100 K: cm^-1, cm^-1, fraction of 300 K (strength: the rule tolerance, 3 %/100 K)
    print("profile_fit slopes per 100 K: 1 fs | 2 fs | |diff| / max(|1 fs slope|, floor)" + ("")
          + f"; floors {FLOOR}; 300 K values compared relatively (positions: |diff| / 1 cm^-1).\nGated: oop and in-plane profile components; complexes (fixed windows) information only.")
    print("Spectra (unpadded, replica mean, matched lags): max |2 fs - 1 fs| / max(1 fs):")
    spec = 0.0
    for T in TS:
        d = np.max(np.abs(b1["2fs"][T]["_s"] - b1["1fs"][T]["_s"])) / b1["1fs"][T]["_s"].max(); spec = max(spec, d)
        print(f"  {T} K: {d:.1e}")
    print("Rule-level band quantities (cation_rules.per_species: what tdep_rules_cation_v1 fits), slope per 100 K and 300 K value:")
    rl = {}
    for n in ("1fs", "2fs"):
        rows = list(csv.DictReader(open(f"{a.work}/{n}_fits.csv")))
        for r in rows:
            r.setdefault("nC", "0")
        getattr(cr, "_A", {}).clear(); rl[n] = cr.per_species(rows)
    rworst = r0worst = 0.0
    for b in sorted(set(rl["1fs"]) | set(rl["2fs"])):
        if b not in rl["1fs"] or b not in rl["2fs"]:
            print(f"  {b:5s} only in the {'1' if b in rl['1fs'] else '2'} fs fit (isolated-component filter)"); continue
        for q in ("position", "width", "strength"):
            s1, _, v1 = rl["1fs"][b][0][1][q]; s2, _, v2 = rl["2fs"][b][0][1][q]
            rel = abs(s2 - s1) / max(abs(s1), FLOOR[q]); rel0 = abs(v2 - v1) / abs(v1); rworst = max(rworst, rel); r0worst = max(r0worst, rel0)
            print(f"  {b:5s} {q:9s} {s1:+8.4f} | {s2:+8.4f} | {rel:.1e} (300 K {v1:.5g}, {rel0:.1e})")
    for k in sorted(fits["1fs"], key=lambda k: (k[0], str(k[1]), str(k[2]))):
        gated = k[0] != "complex"
        if k not in fits["2fs"]:
            lines.append(f"  {k}: missing in the 2 fs fit"); worst = max(worst, 1.0 if gated else 0); continue
        for q in ("position", "width", "strength"):
            s1, v1 = fits["1fs"][k][q]; s2, v2 = fits["2fs"][k][q]
            rel = abs(s2 - s1) / max(abs(s1), FLOOR[q]); rel0 = abs(v2 - v1) / (1.0 if q == "position" else max(abs(v1), 1e-12))
            if gated:
                worst = max(worst, rel, rel0)
            lines.append(f"  {k[0]:7s} {k[1]:>5s} {k[2]!s:>4s} {q:9s} {s1:+8.3f} | {s2:+8.3f} | {rel:.1e} (300 K {v1:.4g}, {rel0:.1e})"
                         + ("" if gated else "  [info]"))
    print("\n".join(lines))
    print(f"spectra: {'PASS' if spec <= 0.01 else 'FAIL'} (largest {spec:.1e}, limit 1e-2)")
    print(f"rule-level band quantities: slopes within {rworst:.1e} (relative to the 1 fs slope or floor), 300 K values within {r0worst:.1e} (relative)")
    print(f"profile components: {'PASS' if worst <= 0.01 else 'FAIL'} (largest {worst:.2e}, limit 1e-2)")

if __name__ == "__main__":
    main()
