# 1 fs vs 2 fs dipole-stride gate of the cation MLMD rules (coordinator 2026-11-26 08:58, item 3), on identical
# dynamics: the gate species' 1 fs dipoles (cation_mlmd_dipoles.py work --gate -> dipoles_gate/<uid>_T<T>_fine_spec.npz)
# and the same series taken every second frame (exactly the 2 fs production frames). Both go through the production
# chain: Rule B1 spectra (ir_sigma, qm=3, unclipped, per-replica T_nve; ACF at FRACTION of the series, 5.1 ps either
# way), cation_spec_export (3x padded grid), profile_fit.fit_species (complexes 3.3/6.2/7.7/8.6, oop and in-plane
# profile components), and per-band T-slopes as profile_fit.per_species. Gate: the oop and in-plane profile components
# (shift, width, strength slopes per 100 K within 1 % of the 1 fs slope, with floors 0.02 cm^-1, 0.01 cm^-1 and 0.03 (the strength rule tolerance); 300 K
# values within 1 %, positions within 0.01 cm^-1). The fixed-window complexes are printed, not gated: their peak is a grid
# argmax and windows gain or lose edge bins (and the coordinator asked for shifting profiles, not fixed windows).
# --long: also the 2 fs series with a 16 ps ACF (FRACTION 0.25 of 64 ps) as a resolution diagnostic (not gated).
#   python cation_gate_stride.py <dipoles_gate dir> <work dir> [--uid 5] [--long]
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlmd_spectra as ms  # noqa: E402
import cation_spec_export as ex  # noqa: E402
import profile_fit as pf  # noqa: E402

TS = (300, 600, 900)


def tree(Z, root, stride, fraction=None):
    for T, z in Z.items():
        sp = str(z["species"]); dt = float(z["dt_fs"]) * stride; out = []
        old = ex.FRACTION
        if fraction:
            ex.FRACTION = fraction
        try:
            for mu, t in zip(z["dipole"], z["T_nve"]):
                out.append(ex.ir_sigma_padded(mu[stride - 1::stride], float(t), dt, 15974.0))
        finally:
            ex.FRACTION = old
        F = z["dipole"].shape[1] // stride; L = int(F * (fraction or ex.FRACTION)) * dt * 1e-15
        os.makedirs(f"{root}/{T}", exist_ok=True)
        np.savez(f"{root}/{T}/{sp}_{T}K.npz", freq=out[0][0], sigma=np.array([s for _, s in out]), prefactor_T=z["T_nve"],
                 replicas=z["replicas"], acf_len_s=L)
        # unpadded B1 integrals per window, as a direct check of the estimator
        yield T, [ms.ir_sigma(mu[stride - 1::stride], float(t), dt_fs=dt, qm=3, clip=False) for mu, t in zip(z["dipole"], z["T_nve"])] if not fraction else None
    return


def b1_windows(spec):
    f = spec[0][0]; s = np.mean([x[1] for x in spec], 0); df = f[1] - f[0]; tot = s.sum() * df
    return {b: s[(f >= lo) & (f < hi)].sum() * df / tot for b, lo, hi in pf.BANDS} | {"total": tot}


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
    ap.add_argument("--long", action="store_true")
    a = ap.parse_args()
    Z = {T: dict(np.load(f"{a.gate}/{a.uid}_T{T}_fine_spec.npz")) for T in TS}
    sp = str(Z[300]["species"])
    print(f"{sp}: {[Z[T]['dipole'].shape for T in TS]} at {float(Z[300]['dt_fs'])} fs; T_nve " +
          ", ".join(f"{T}: {np.mean(Z[T]['T_nve']):.0f}" for T in TS))
    runs = {"1fs": (1, None), "2fs": (2, None)} | ({"2fs_L16": (2, 0.25)} if a.long else {})
    fits, b1 = {}, {}
    for name, (stride, frac) in runs.items():
        root = f"{a.work}/{name}"
        b1[name] = {T: b1_windows(s) for T, s in tree(Z, root, stride, frac) if s is not None}
        rows = pf.fit_species(sp, {T: f"{root}/{T}" for T in TS}, [("oop", 650, 1000), ("inplane", 1100, 1700)], complexes=True)
        with open(f"{a.work}/{name}_fits.csv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        fits[name] = slopes(rows)
    worst = 0.0; lines = []
    print("B1 fixed-window fractions (unpadded ir_sigma, replica mean), 2 fs / 1 fs - 1; information only: the two DCT grids differ\n"
          "slightly (3.264 vs 3.265 cm^-1), so a window gains or loses an edge bin, and 2 fs uses half the ACF time origins:")
    for T in TS:
        d = {b: b1["2fs"][T][b] / b1["1fs"][T][b] - 1 for b in b1["1fs"][T] if b1["1fs"][T][b] > 1e-3 * (b != "total") }
        print(f"  {T} K: " + ", ".join(f"{b} {v:+.1e}" for b, v in d.items()))
    FLOOR = {"position": 0.02, "width": 0.01, "strength": 0.03}      # per 100 K: cm^-1, cm^-1, fraction of 300 K (strength: the rule tolerance, 3 %/100 K)
    print("profile_fit slopes per 100 K: 1 fs | 2 fs | |diff| / max(|1 fs slope|, floor)" + (" | 2 fs with a 16 ps ACF (component nearest in 300 K centre)" if a.long else "")
          + f"; floors {FLOOR}; 300 K values compared relatively (positions: |diff| / 1 cm^-1).\nGated: oop and in-plane profile components; complexes (fixed windows) information only.")
    def near(k, q):
        c = [(abs(v["position"][1] - fits["1fs"][k]["position"][1]), v) for kk, v in fits.get("2fs_L16", {}).items() if kk[0] == k[0]]
        c = min(c, key=lambda t: t[0], default=None)
        return f" | {c[1][q][0]:+.3f}" if c and c[0] < 3 else " | -"
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
                         + (near(k, q) if a.long else "") + ("" if gated else "  [info]"))
    print("\n".join(lines))
    print(f"GATE {'PASS' if worst <= 0.01 else 'FAIL'}: largest gated difference {worst:.2e} (limit 1e-2)")

if __name__ == "__main__":
    main()
