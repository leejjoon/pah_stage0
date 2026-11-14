# Overlap analysis (plan d1_hangang_new_species.md, step 3): our MLMD spectra of the overlap species against Mai et
# al.'s published ones (Zenodo 15771437, outputs/IR_txt_phdb_1704/IR_txt_<T>K_qm0.zip), and the scatter between
# our replicas, which sets the replica count of later runs. Generalises port/mai_spectrum_compare.py (naphthalene).
# One trajectory is noisy, so compare per band window, not pointwise: each window's share of the 300-3800 cm^-1
# intensity and its intensity-weighted centroid. Systems that fail the screen or the dipole check are left out.
#   python overlap_compare.py <spectra dir> <run dir> <Supplementary.zip> <overlap_species.csv> <out.csv>
# out.csv: one row per species, T and window. Prints the summary: how often the published share lies inside our
# replica range (expected (N-1)/(N+1) if both come from the same distribution), centroid offsets, and the replica
# scatter of the shares against Nc with the replica count needed for a given precision.
import sys, io, csv, zipfile, collections
import numpy as np
trap = getattr(np, "trapezoid", None) or np.trapz

WINDOWS = [("CH oop 650-950", 650, 950), ("CC/CH ip 1100-1500", 1100, 1500), ("CC str 1500-1700", 1500, 1700),
           ("CH str 2950-3150", 2950, 3150), ("low 300-650", 300, 650), ("950-1100", 950, 1100)]


def shares(x, y):
    tot = trap(y, x); out = []
    for _, lo, hi in WINDOWS:
        s = (x >= lo) & (x < hi); xs, ys = x[s], y[s]
        out.append((trap(ys, xs) / tot, (xs * ys).sum() / ys.sum() if ys.sum() > 0 else np.nan))
    return np.array(out)                                             # (window, [share, centroid])


def main():
    spec, run, supp, species_csv, out_csv = sys.argv[1:6]
    bad = {(r["species"], float(r["T"]), int(r["replica"])) for r in csv.DictReader(open(f"{run}/screen.csv"))
           if r["pass"] != "True"}
    bad |= {(r["species"], float(r["T"]), int(r["replica"])) for r in csv.DictReader(open(f"{spec}/qa.csv"))
            if r["shift_flag"] == "1"}
    nC = {r["species"]: int(r["nC"]) for r in csv.DictReader(open(species_csv))}
    z = zipfile.ZipFile(supp); rows = []; summ = collections.defaultdict(list)
    for T in (300, 600):
        inner = zipfile.ZipFile(io.BytesIO(z.read(f"Supplementary_Data_Code/outputs/IR_txt_phdb_1704/IR_txt_{T}K_qm0.zip")))
        for sp in sorted(nC):
            P = shares(*np.loadtxt(io.StringIO(inner.read(f"IR_txt_{T}K_qm0/{sp}.txt").decode()), unpack=True))
            d = np.load(f"{spec}/{sp}_{T}K.npz")
            keep = [i for i, r in enumerate(d["replicas"]) if (sp, float(T), int(r)) not in bad]
            if len(keep) < 3:
                summ["skipped"].append((sp, T, len(keep))); continue
            R = np.array([shares(d["freq"], d["sigma"][i]) for i in keep])   # (rep, window, 2)
            M = shares(d["freq"], d["sigma"][keep].mean(0))
            for w, (name, _, _) in enumerate(WINDOWS):
                sh = R[:, w, 0]; n = len(sh)
                inside = sh.min() <= P[w, 0] <= sh.max()
                rows.append(dict(species=sp, T=T, nC=nC[sp], window=name, n_rep=n, pub_share=P[w, 0],
                                 rep_share_mean=sh.mean(), rep_share_sd=sh.std(ddof=1), rep_share_min=sh.min(),
                                 rep_share_max=sh.max(), pub_inside=int(inside), pub_centroid=P[w, 1],
                                 mean_centroid=M[w, 1], centroid_offset=P[w, 1] - M[w, 1]))
                summ[(T, name)].append(rows[-1])
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"{len(rows)} rows -> {out_csv}; skipped (fewer than 3 clean replicas): {summ['skipped']}")
    print("Published vs our replicas, per T and window (share = window's fraction of 300-3800 cm-1 intensity):")
    print(f"{'T':>4s} {'window':20s} {'n':>3s} {'inside':>7s} {'expect':>6s} {'pub-ours share':>15s} {'centroid off':>13s} "
          f"{'rel. sd 1 rep':>13s}")
    for T in (300, 600):
        for name, _, _ in WINDOWS:
            v = summ[(T, name)]
            if not v: continue
            n = np.median([r["n_rep"] for r in v])
            d_sh = np.array([r["pub_share"] - r["rep_share_mean"] for r in v])
            off = np.array([r["centroid_offset"] for r in v])
            rel = np.array([r["rep_share_sd"] / r["rep_share_mean"] for r in v if r["rep_share_mean"] > 0])
            print(f"{T:4d} {name:20s} {len(v):3d} {np.mean([r['pub_inside'] for r in v]):7.2f} {(n - 1) / (n + 1):6.2f} "
                  f"{np.median(d_sh):+7.4f} ±{np.median(np.abs(d_sh)):.4f} {np.median(off):+6.1f} ±{np.median(np.abs(off)):4.1f} "
                  f"{np.median(rel):13.3f}")
    print("Replica scatter of the shares vs Nc (median relative sd of one replica, strong windows; the sd of a mean of"
          " n replicas is this / sqrt(n)):")
    strong = ("CH oop 650-950", "CC/CH ip 1100-1500", "CC str 1500-1700", "CH str 2950-3150")
    for T in (300, 600):
        for lo, hi in ((0, 30), (30, 60), (60, 100), (100, 999)):
            rel = [r["rep_share_sd"] / r["rep_share_mean"] for name in strong for r in summ[(T, name)]
                   if lo <= r["nC"] < hi and r["rep_share_mean"] > 0]
            if rel:
                m = np.median(rel)
                print(f"   {T} K, Nc {lo}-{hi}: {len(rel) // len(strong):2d} species, rel. sd {m:.3f}; "
                      f"n for 2 %: {int(np.ceil((m / 0.02) ** 2))}, for 5 %: {int(np.ceil((m / 0.05) ** 2))}")


if __name__ == "__main__":
    main()
