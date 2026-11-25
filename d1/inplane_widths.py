# Per-line widths of isolated in-plane bands (1100-1700 cm^-1) from `profile_fit.py fit --regions inplane` outputs.
# A component counts as isolated if, in the replica-mean fit at each of 300, 600 and 900 K, the nearest other
# component is farther away than SEP x the sum of the two FWHMs and its FWHM is off the fit bounds (0.5-50 cm^-1), and it carries at least MIN_AREA of the total intensity at
# 300 K. Per isolated band: FWHM at 300 K and the slope of FWHM against the replicas' NVE T. Families by 300 K centre.
#   python inplane_widths.py <fits csv> [more csv ...] [--keep <file with one species per line>]
# --keep restricts the fit to the listed species (perline_widths_clean.py: the 385 spin-consistent closed-shell singlets).
import csv, sys, collections
import numpy as np

MIN_AREA = 0.003
SEP = 0.75        # 1.0 left 2 of 60 components in a 5-species test; the 900 K fits are partly degenerate
FAMILIES = [("CC stretch (6.2 um)", 1540, 1700), ("1470-1540", 1470, 1540), ("CC/CH in-plane (7.7 um)", 1220, 1470),
            ("CH in-plane (8.6 um)", 1100, 1220)]


def collect(files, keep=None):
    """{family: [(nC, FWHM at 300 K, slope per 100 K, fit rms)]} of the isolated bands, species count, component count."""
    R = [r for f in files for r in csv.DictReader(open(f)) if r["region"] == "inplane" and (keep is None or r["species"] in keep)]
    mean = collections.defaultdict(dict)                     # (species, T) -> comp -> (centre, fwhm, area)
    reps = collections.defaultdict(list)                     # (species, comp) -> replica rows
    for r in R:
        if r["replica"] == "mean":
            mean[(r["species"], r["T"])][r["comp"]] = (float(r["center"]), float(r["fwhm"]), float(r["area"]))
        else:
            reps[(r["species"], r["comp"])].append(r)
    out = collections.defaultdict(list); n_all = 0
    for (sp, comp), v in reps.items():
        n_all += 1
        ok = True
        for T in ("300", "600", "900"):
            cs = mean.get((sp, T), {})
            if comp not in cs:
                ok = False; break
            c, fw, _ = cs[comp]
            if not 0.5 < fw < 50 or any(abs(c - c2) < SEP * (fw + fw2) for k, (c2, fw2, _) in cs.items() if k != comp):
                ok = False; break
        if not ok or mean[(sp, "300")][comp][2] < MIN_AREA:
            continue
        c300 = mean[(sp, "300")][comp][0]
        fam = next((n for n, lo, hi in FAMILIES if lo <= c300 < hi), None)
        if fam is None:
            continue
        x = np.array([float(r["T_nve"]) for r in v]) / 100; y = np.array([float(r["fwhm"]) for r in v])
        out[fam].append((int(v[0]["nC"]), np.mean([float(r["fwhm"]) for r in v if r["T"] == "300"]), np.polyfit(x, y, 1)[0],
                         np.median([float(r["rms"]) for r in v])))
    return out, len({k[0] for k in reps}), n_all


def main(files, keep=None):
    out, n_sp, n_all = collect(files, keep)
    print(f"{n_sp} species, {n_all} components; isolated at all three T and >= {MIN_AREA:.1%} of the "
          f"total at 300 K (separation >= {SEP} x sum of FWHMs), per family: per-line FWHM at 300 K and its slope per 100 K, median [16-84 %]")
    for fam, *_ in FAMILIES:
        v = np.array(out.get(fam, []))
        if len(v) < 5:
            print(f"  {fam:26s} {len(v)} bands"); continue
        X = np.log10(v[:, 0] / 50); c = np.polyfit(X, v[:, 2], 1)
        print(f"  {fam:26s} {len(v):4d} bands from {len(set(v[:, 0]))} sizes | FWHM 300 K {np.median(v[:, 1]):5.2f} "
              f"[{np.percentile(v[:, 1], 16):.2f}..{np.percentile(v[:, 1], 84):.2f}] | slope {np.median(v[:, 2]):+.2f} "
              f"[{np.percentile(v[:, 2], 16):+.2f}..{np.percentile(v[:, 2], 84):+.2f}] per 100 K (fit at Nc 50 {c[1]:+.2f}, "
              f"{c[0]:+.2f}/dex) | fit rms {np.median(v[:, 3]):.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]; keep = None
    if "--keep" in a:
        i = a.index("--keep"); keep = set(open(a[i + 1]).read().split()); a = a[:i] + a[i + 2:]
    main(a, keep)
