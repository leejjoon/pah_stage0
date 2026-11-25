# Finite-T IR band strengths, peak shifts, widths and temperature slopes from replica MD dipole series, scored the same
# way for the cloud-aimd DFT MD (d1/aimd_pyscf.py, 1 fs dipoles) and the MLMD of d1/mlmd_polar.py (2 fs), for the
# coordinator's 2026-11-24 17:03 feat-dipole steps 2 (naphthalene vs cloud-aimd, Gate B4 pilot) and 3 (large cations).
#   python mlmd_tslopes.py <ref.npz> <run.npz ...> [--label X] [--nve-ps-max 20] [--dt-fs 2] [--offset-D 1] [--json out]
# Per run (as d1/aimd_slopes.py): d1/mlmd_spectra.ir_sigma, qm=3, unclipped, the run's mean NVE temperature in the
# prefactor; band strength = integral (km/mol) over windows of the scaled axis. --nve-ps-max / --dt-fs truncate and
# subsample every series to a common length and step, so line shapes (the 7.987 % autocorrelation window) compare.
# Per nominal T: strengths (replica mean +- s.e.), the replica-mean spectrum's peak position and FWHM in each window
# (shift and width), and the mode-normalised strength of d1/mode_energy.py (measured / harmonic x F, F = sum_W I_k
# E_k/kT / sum_W I_k with E_k from the saved NVE frames, ref = the harmonic reference: mode_energy.py ref for DFT,
# mlmd_polar.py ref for the MLMD). Slopes in % of the lowest-T value per 100 K: raw (straight line through all
# replicas, fraction of the fitted lowest-T value) and normalised (mode_energy.py: lowest vs highest T group means;
# with three temperatures also a straight-line fit through the group means).
# --offset-D: static-offset test. A body-fixed constant dipole error of that size (D) along each principal axis of the
# reference, rotated with the molecule (orientation from a mass-weighted Kabsch fit of each saved frame to the reference,
# interpolated to the dipole times), is added to the dipole series; reported: the largest relative change of any band
# strength and of the raw slopes. A static error that enters only as mu_0 drops out of <mu_dot(0) mu_dot(t)>, but the
# estimator here transforms <mu(0) mu(t)>, so this tests what the production estimator actually does with it.
import argparse, collections, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma, SCALE  # noqa: E402
from mode_energy import mode_E, kabsch, KB_EV  # noqa: E402

KM = 6.02214076e23 * 1e-5
D_PER_EA = 4.80320471
WIN = {"oop": (600, 1000), "1000-1300": (1000, 1300), "1300-1700": (1300, 1700), "inplane": (1000, 1700),
       "CHstr": (2700, 3300), "12.7": (757.6, 806.5), "11.2": (862.1, 925.9), "8.6": (1123.6, 1219.5),
       "7.7": (1219.5, 1428.6), "6.2": (1538.5, 1666.7), "3.3": (2777.8, 3225.8)}


def series(z, a):
    mu = np.asarray(z["dipole"], float); dt = float(z["dt_fs"])
    ok = np.isfinite(mu).all(1); mu = mu[:ok.argmin()] if not ok.all() else mu
    k = max(1, int(round(a.dt_fs / dt))) if a.dt_fs else 1
    mu = mu[k - 1::k]; dt = dt * k
    if a.nve_ps_max:
        mu = mu[:int(round(a.nve_ps_max * 1000 / dt))]
    return mu, dt


def nve_T(z):
    qa = np.asarray(z["qa"]); n_nvt = int(round(float(z["nvt_ps"]) * 2000))
    return float(qa[qa[:, 0] > n_nvt][:, 1].mean())


def spectrum(mu, T, dt):
    f, s = ir_sigma(mu, T, dt, 3, False)
    return f, s


def strengths(f, s):
    df = (f[1] - f[0]) / SCALE
    return {w: float(s[(f >= lo) & (f < hi)].sum() * df * KM) for w, (lo, hi) in WIN.items()}


def peak(f, s, lo, hi):
    m = (f >= lo) & (f < hi); ff, ss = f[m], s[m]
    k = int(np.argmax(ss)); h = ss[k] / 2
    i = k
    while i > 0 and ss[i] > h:
        i -= 1
    j = k
    while j < len(ss) - 1 and ss[j] > h:
        j += 1
    return float(ff[k]), float(ff[j] - ff[i])


def orient(rf, z, n_mu, dt_mu):
    """body->lab rotation at each dipole time, from the saved frames (Kabsch to the reference), linear in time between
    frames and re-orthonormalised"""
    m = rf["masses"]; P = np.asarray(z["frames_pos"], float); every_fs = int(z["frames_every"]) * 0.5
    Rs = []
    for p in P:
        c = p - (m[:, None] * p).sum(0) / m.sum()
        Rs.append(kabsch(c, rf["pos"], m).T)          # kabsch gives lab->ref; transpose: ref(body)->lab
    Rs = np.array(Rs); tf = (np.arange(len(Rs)) + 1) * every_fs; tm = (np.arange(n_mu) + 1) * dt_mu
    out = np.empty((n_mu, 3, 3))
    for i, t in enumerate(tm):
        k = np.clip(np.searchsorted(tf, t), 1, len(tf) - 1); w = np.clip((t - tf[k - 1]) / (tf[k] - tf[k - 1]), 0, 1)
        M = (1 - w) * Rs[k - 1] + w * Rs[k]; U, _, Vt = np.linalg.svd(M); out[i] = U @ Vt
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--label", default=""); ap.add_argument("--nve-ps-max", type=float, default=0)
    ap.add_argument("--dt-fs", type=float, default=0); ap.add_argument("--offset-D", type=float, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    rf = dict(np.load(a.ref)); wn_s = rf["wn"] * SCALE; I = rf["I"]
    # principal axes of the reference (body frame) for the offset test
    m = rf["masses"]; X = rf["pos"]; It = (m[:, None, None] * (np.einsum("ij,ik->ijk", X, X))).sum(0)
    axes = np.linalg.eigh(np.trace(It) * np.eye(3) - It)[1].T
    runs = collections.defaultdict(list)
    for p in a.runs:
        z = np.load(p); Tn = int(re.search(r"_(\d+)K_r\d+\.npz$", p).group(1))
        mu, dt = series(z, a); T = nve_T(z)
        f, s = spectrum(mu, T, dt); st = strengths(f, s)
        r = dict(run=os.path.basename(p), T=T, dt=dt, n=len(mu), f=f, s=s, strength=st)
        if "frames_pos" in z.files and len(z["frames_pos"]):
            E = np.array([mode_E(rf, P, V)[0] for P, V in zip(z["frames_pos"], z["frames_vel"])]).mean(0)
            rr = E / (KB_EV * T); r["norm"] = {}
            for w, (lo, hi) in WIN.items():
                sel = (wn_s >= lo) & (wn_s < hi)
                if I[sel].sum() > 0:
                    F = (I[sel] * rr[sel]).sum() / I[sel].sum(); r["norm"][w] = st[w] / (I[sel].sum() * F)
            if a.offset_D:
                R = orient(rf, z, len(mu), dt); r["offset"] = {}
                for ax in range(3):
                    off = np.einsum("tij,j->ti", R, axes[ax]) * a.offset_D / D_PER_EA
                    r["offset"][ax] = strengths(*spectrum(mu + off, T, dt))
        runs[Tn].append(r)
    Ts = sorted(runs); out = dict(label=a.label, temps={}, slopes={})
    print(f"=== {a.label}: {len(a.runs)} runs, T groups " + ", ".join(f"{T} K x{len(runs[T])}" for T in Ts)
          + f"; series {runs[Ts[0]][0]['n']} x {runs[Ts[0]][0]['dt']:.0f} fs")
    print(f"{'window':10s} {'harm':>7s} " + "".join(f"| {T:4d} K: strength km/mol  norm   peak  FWHM " for T in Ts))
    for w in WIN:
        cells = []; sel = (wn_s >= WIN[w][0]) & (wn_s < WIN[w][1]); harm = I[sel].sum()
        for T in Ts:
            R = runs[T]; v = np.array([r["strength"][w] for r in R]); se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            nv = np.array([r["norm"][w] for r in R if "norm" in r and w in r["norm"]])
            fm = R[0]["f"]; sm = np.mean([np.interp(fm, r["f"], r["s"]) for r in R], 0)
            pk, fw = peak(fm, sm, *WIN[w])
            out["temps"].setdefault(str(T), {})[w] = dict(T=float(np.mean([r["T"] for r in R])), strength=float(v.mean()), se=float(se),
                                                       norm=float(nv.mean()) if len(nv) else None,
                                                       norm_se=float(nv.std(ddof=1) / np.sqrt(len(nv))) if len(nv) > 1 else None,
                                                       peak=pk, fwhm=fw, harmonic=float(harm))
            cells.append(f"| {v.mean():8.1f} +- {se:5.1f}  " + (f"{nv.mean():5.3f}" if len(nv) else "  -  ") + f" {pk:6.0f} {fw:5.0f} ")
        print(f"{w:10s} {harm:7.1f} " + "".join(cells))
    print("slopes, % per 100 K of the lowest-T value: raw (all replicas) | normalised lo-hi +- s.e. | normalised fit | "
          "peak shift cm^-1/100 K | FWHM change cm^-1/100 K" + (f" | max |raw slope change| with a {a.offset_D} D body-fixed offset" if a.offset_D else ""))
    for w in WIN:
        allr = [r for T in Ts for r in runs[T]]
        x = np.array([r["T"] for r in allr]) / 100; y = np.array([r["strength"][w] for r in allr])
        b1, b0 = np.polyfit(x, y, 1); x0 = min(x); raw = 100 * b1 / (b1 * x0 + b0)
        tt = out["temps"]; lo, hi = str(Ts[0]), str(Ts[-1])
        o = dict(raw_pct=float(raw))
        line = f"  {w:10s} raw {raw:+6.2f}"
        if tt[lo][w]["norm"] is not None and tt[hi][w]["norm"] is not None:
            m0, e0, T0 = tt[lo][w]["norm"], tt[lo][w]["norm_se"] or 0, tt[lo][w]["T"]
            m1, e1, T1 = tt[hi][w]["norm"], tt[hi][w]["norm_se"] or 0, tt[hi][w]["T"]
            dT = (T1 - T0) / 100; s = 100 * (m1 / m0 - 1) / dT; es = 100 * np.hypot(e1 / m0, m1 * e0 / m0 ** 2) / dT
            o.update(norm_pct=float(s), norm_se_pct=float(es)); line += f" | norm {s:+6.2f} +- {es:4.2f}"
            if len(Ts) > 2:
                xs = np.array([tt[str(T)][w]["T"] for T in Ts]) / 100; ys = np.array([tt[str(T)][w]["norm"] for T in Ts])
                c1, c0 = np.polyfit(xs, ys, 1); fit = 100 * c1 / (c1 * xs[0] + c0); o["norm_fit_pct"] = float(fit)
                line += f" | fit {fit:+6.2f}"
        xs = np.array([tt[str(T)][w]["T"] for T in Ts]) / 100
        ps = np.polyfit(xs, [tt[str(T)][w]["peak"] for T in Ts], 1)[0]; ws = np.polyfit(xs, [tt[str(T)][w]["fwhm"] for T in Ts], 1)[0]
        o.update(peak_shift_per_100K=float(ps), fwhm_change_per_100K=float(ws))
        line += f" | shift {ps:+5.2f} | FWHM {ws:+5.2f}"
        if a.offset_D:
            offr = [r for r in allr if "offset" in r]
            if offr:
                xo = np.array([r["T"] for r in offr]) / 100; d_raw, d_str = 0.0, 0.0
                for ax in range(3):
                    yo = np.array([r["offset"][ax][w] for r in offr]); y0 = np.array([r["strength"][w] for r in offr])
                    q1, q0 = np.polyfit(xo, yo, 1); p1, p0 = np.polyfit(xo, y0, 1)
                    d_raw = max(d_raw, abs(100 * q1 / (q1 * min(xo) + q0) - 100 * p1 / (p1 * min(xo) + p0)))
                    d_str = max(d_str, float(np.max(np.abs(yo / y0 - 1))))
                o.update(offset_raw_slope_change_pct=d_raw, offset_max_rel_strength_change=d_str)
                line += f" | offset: slope {d_raw:.3f} %/100 K, strength {100 * d_str:.2f} %"
        out["slopes"][w] = o; print(line)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
