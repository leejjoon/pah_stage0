# Energy along G4 failure paths (frames from g4_fail_frames.py, labelled by dft_seed_run.py), DFT vs MACE models,
# container B:
#   python g4_path_energy.py <dft.extxyz> <out.json> --models v4=<model>,v7=<model> [--bs-dir <dft_bs dir>] [--device cuda]
# Per path (one MD run): every frame's energy relative to the path's frame 0 (the relaxed MD start), DFT and each
# model, in eV, with the bond stretch s (max d/cutoff over the bonds that change; > 1 = broken), the frame's MD time,
# and each model's force MAE (meV/A). --bs-dir: the broken-symmetry UKS energy (dft_seed_run.py run --stab-all --bs)
# where it lies below RKS by > 1e-6 Ha; a homolysis on the closed-shell (q0) surface is a biradical, which RKS puts too
# high. Answers: does DFT have a low path to the change the model made, and how far off is the model along it.
import argparse, collections, glob, json, os
import numpy as np

HARTREE_EV = 27.211386245988


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dft"); ap.add_argument("out"); ap.add_argument("--models", required=True)
    ap.add_argument("--bs-dir", default=None); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from ase.io import read
    from mace.calculators import MACECalculator
    ats = read(a.dft, ":"); models = dict(x.split("=", 1) for x in a.models.split(","))
    bs = {}
    for f in glob.glob(f"{a.bs_dir}/*.json") if a.bs_dir else []:
        r = json.load(open(f))
        if r.get("bs_guess") and r.get("bs_converged") and r.get("bs_dE_Ha", 0) > 1e-6:
            bs[os.path.basename(f)[:-5]] = (r["bs_energy_Ha"] * HARTREE_EV, r.get("bs_s2"))
    paths = collections.defaultdict(list)
    for at in ats:
        paths[at.info["log"].rsplit("_f", 1)[0]].append(at)
    pred = {}
    for m, p in models.items():
        calc = {q: MACECalculator(model_paths=p, device=a.device, default_dtype="float32", head=f"q{q}") for q in (0, 1)}
        for at in ats:
            b = at.copy(); b.calc = calc[int(at.info["charge"])]
            pred[m, at.info["log"]] = (b.get_potential_energy(),
                                       1000 * float(np.abs(b.get_forces() - at.arrays["dft_forces"]).mean()))
    out = {}
    for name, fr in sorted(paths.items()):
        fr.sort(key=lambda x: int(x.info["log"].rsplit("_f", 1)[1]))
        ref = fr[0]; rows = []
        print(f"\n### {name}: energy relative to frame 0, eV (s = bond stretch, > 1 broken; F = force MAE meV/A)")
        print("| frame | MD ps | s | DFT | " + ("BS-UKS (<S^2>) | " if bs else "") + " | ".join(f"{m} | F {m}" for m in models) + " |")
        for at in fr:
            k = at.info["log"]; e = at.info["dft_energy"] - ref.info["dft_energy"]
            row = dict(frame=k, md_ps=float(at.info.get("md_ps", 0)), s=float(at.info["max_stretch"]),
                       graph_changed=int(at.info["graph_changed"]), dft=e, stability=at.info.get("stability"), s2=float(at.info.get("s2", 0)))
            if k in bs:
                row["bs"] = bs[k][0] - ref.info["dft_energy"]; row["bs_s2"] = bs[k][1]
            for m in models:
                row[m] = pred[m, k][0] - pred[m, ref.info["log"]][0]; row[f"F_{m}"] = pred[m, k][1]
            rows.append(row)
            bcol = (f"{row['bs']:.2f} ({row['bs_s2']:.2f}) | " if "bs" in row else "– | ") if bs else ""
            print(f"| {k.rsplit('_f', 1)[1]} | {row['md_ps']:.2f} | {row['s']:.2f}{'*' if row['graph_changed'] else ''} | {e:.2f} | {bcol}"
                  + " | ".join(f"{row[m]:.2f} | {row['F_' + m]:.0f}" for m in models) + " |")
        out[name] = rows
    json.dump(dict(models=models, paths=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
