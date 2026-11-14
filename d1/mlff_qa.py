# Data QA for the charged-state MLFF DFT data (committee review M2, gate G1; docs/plan/charged_mlff_plan.md).
# Nothing is dropped: every frame is written back with a qa_flags label, and a quarantine report lists the reasons.
#   python mlff_qa.py <parsed.extxyz> <out prefix> [--manifest gjf/manifest.json] [--no-xtb]
# Checks, per species and charge (the frames of one sampling run, "group"):
#   missing          manifest entries with no parsed result (failed jobs; listed in <prefix>_missing.txt)
#   unstable         stability test found an instability that was not re-optimised (closed-shell RKS->UKS, or internal)
#   s2_review        open shell with <S^2> before annihilation > 0.80, or > group median + 0.05 (a trigger for review,
#                    not a certificate either way)
#   energy_outlier   E - E(min frame) disagrees with GFN2-xTB's E - E(min frame) on the same geometries by more than
#                    max(6 robust sigma of the group's residuals, 0.5 eV): the SCF likely landed on another state
#   dipole_outlier   |mu_DFT - mu_xTB| (about the centre of mass) beyond max(6 robust sigma, 0.3 e*A) of the group
# Informational (not quarantined): reoptimized (Stable=Opt changed the wavefunction, result is stable),
# persistent_change (MD frame whose bond graph change lasted 100 fs), graph_changed, boundary (bound_* frames).
# Writes <prefix>.extxyz (all frames, qa_flags added, plus xtb_energy / xtb_dipole), <prefix>_report.csv (one row per
# flagged frame) and prints a summary. xTB needs the micromamba env /var/tmp/$USER-pah/mamba/xtb (one core).
import argparse, collections, csv, json, os, re, sys
import numpy as np

QUARANTINE = {"unstable", "s2_review", "energy_outlier", "dipole_outlier"}
MASS = {"H": 1.00782503, "C": 12.0, "N": 14.00307401, "O": 15.99491462}


def read(path):
    out = []
    with open(path) as f:
        while (line := f.readline()).strip():
            n = int(line); head = f.readline(); info = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', head))
            rows = [f.readline().split() for _ in range(n)]
            out.append(dict(info={k: v.strip('"') for k, v in info.items()}, head=head.rstrip("\n"), rows=rows))
    return out


def robust_sigma(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else 0.0


def xtb_values(fr):
    from ase import Atoms
    from tblite.ase import TBLite
    sym = [r[0] for r in fr["rows"]]; p = np.array([[float(x) for x in r[1:4]] for r in fr["rows"]])
    q, m = int(fr["info"]["charge"]), int(fr["info"]["mult"])
    at = Atoms(sym, positions=p); at.calc = TBLite(method="GFN2-xTB", charge=q, multiplicity=m, verbosity=0)
    e = at.get_potential_energy(); mu = at.get_dipole_moment()
    mass = np.array([MASS[s] for s in sym]); com = (mass[:, None] * p).sum(0) / mass.sum()
    return e, mu - q * com


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed"); ap.add_argument("prefix")
    ap.add_argument("--manifest", default=None, help="gjf/manifest.json for the missing-job check ('none' to skip)"); ap.add_argument("--no-xtb", action="store_true")
    a = ap.parse_args()
    frames = read(a.parsed)
    for fr in frames:
        fr["flags"] = []
        i = fr["info"]
        fr["name"] = i.get("log", "?").rsplit(".", 1)[0]
        fr["group"] = (i.get("source", "?"), i.get("charge", "?"))
        if i.get("stability") in ("unstable_internal", "unstable_rhf_uhf"):
            fr["flags"].append("unstable")
        if i.get("stability") == "reoptimized":
            fr["flags"].append("reoptimized")
        if i.get("changed_n") == "3":
            fr["flags"].append("persistent_change")
        if i.get("graph_changed") == "1":
            fr["flags"].append("graph_changed")
        if i.get("config_type", "").startswith("bound"):
            fr["flags"].append("boundary")
    groups = collections.defaultdict(list)
    for fr in frames:
        groups[fr["group"]].append(fr)
    for g, fs in groups.items():
        if int(fs[0]["info"].get("mult", 1)) > 1:
            s2 = np.array([float(f["info"].get("s2", 0)) for f in fs]); med = np.median(s2)
            for f, v in zip(fs, s2):
                if v > 0.80 or v > med + 0.05:
                    f["flags"].append("s2_review")
        if a.no_xtb:
            continue
        for f in fs:
            f["xtb_E"], f["xtb_mu"] = xtb_values(f)
        ref = next((f for f in fs if f["info"].get("config_type") == "min"), None) or min(fs, key=lambda f: float(f["info"]["dft_energy"]))
        res = np.array([(float(f["info"]["dft_energy"]) - float(ref["info"]["dft_energy"])) - (f["xtb_E"] - ref["xtb_E"]) for f in fs])
        dmu = np.array([np.linalg.norm(np.array([float(x) for x in f["info"]["dft_dipole"].split()]) - f["xtb_mu"]) for f in fs])
        for arr, flag, floor in ((res, "energy_outlier", 0.5), (dmu, "dipole_outlier", 0.3)):
            lim = max(6 * robust_sigma(arr), floor); med = np.median(arr)
            for f, v in zip(fs, arr):
                f.setdefault("qa", {})[flag] = v - med
                if abs(v - med) > lim:
                    f["flags"].append(flag)
    missing = []
    mp = a.manifest or os.path.join(os.path.dirname(os.path.abspath(a.parsed)), "gjf", "manifest.json")
    if a.manifest != "none" and os.path.exists(mp) and os.path.getsize(mp):
        have = {f["name"] for f in frames}
        missing = sorted(k for k in json.load(open(mp)) if k not in have)
        open(f"{a.prefix}_missing.txt", "w").write("\n".join(missing) + ("\n" if missing else ""))
    with open(f"{a.prefix}.extxyz", "w") as out:
        for f in frames:
            extra = f" qa_flags={','.join(f['flags']) or 'ok'}"
            if "xtb_E" in f:
                extra += f" xtb_energy={f['xtb_E']:.6f} xtb_dipole=\"{' '.join(f'{x:.5f}' for x in f['xtb_mu'])}\""
            out.write(f"{len(f['rows'])}\n{f['head']}{extra}\n")
            out.writelines(" ".join(r) + "\n" for r in f["rows"])
    with open(f"{a.prefix}_report.csv", "w", newline="") as rep:
        w = csv.writer(rep); w.writerow(["frame", "source", "charge", "config_type", "sample_T", "flags", "quarantine",
                                         "s2", "stability", "energy_residual_eV", "dipole_residual_eA"])
        for f in frames:
            if f["flags"]:
                i = f["info"]; qa = f.get("qa", {})
                w.writerow([f["name"], i.get("source"), i.get("charge"), i.get("config_type"), i.get("sample_T"),
                            ";".join(f["flags"]), int(bool(QUARANTINE & set(f["flags"]))), i.get("s2"), i.get("stability"),
                            f"{qa.get('energy_outlier', float('nan')):.3f}", f"{qa.get('dipole_outlier', float('nan')):.3f}"])
    c = collections.Counter(fl for f in frames for fl in f["flags"])
    nq = sum(bool(QUARANTINE & set(f["flags"])) for f in frames)
    print(f"{len(frames)} frames in {len(groups)} groups; quarantined {nq}; missing {len(missing)}")
    for k, v in sorted(c.items()):
        print(f"  {k:18s} {v}{'  (quarantine)' if k in QUARANTINE else ''}")


if __name__ == "__main__":
    main()
