# Dipole half of G2 (docs/plan/charged_mlff_proposal.md §5.7: dipole MAE <= 0.06 D neutrals, <= 0.1 D ions about the
# centre of mass) for dipole-only mace-torch models, one model per charge: AtomicDipolesMACE (d1/mace_torch_dipole.yaml,
# --model-type DipoleMACE, the default) or the charge-constrained AtomicDielectricMACE (d1/mace_torch_dielectric.yaml,
# --model-type DipolePolarizabilityMACE). Container B or any env with mace-torch 0.3.17.
#   python mlff_dipole_eval.py <dataset dir> <out.json> --models q0=<model>,q1=<model> [--device cuda] [--model-type ..]
# Every geometry is moved to its centre of mass (isotopic masses, g16_io.MASS, as the label) before prediction: the
# dielectric model's sum q_i r_i depends on the origin for ions and was trained on COM-centred data
# (d1/mlff_dipole_prep.py); for the translation-invariant AtomicDipolesMACE centring changes nothing.
# PolarMACE (d1/mace_torch_polar.yaml, --model-type PolarMACE): one model for both charges, so pass the same path
# twice (--models q0=<m>,q1=<m>); charge and spin multiplicity come from the frame's info keys charge and mult.
# --add-xtb: Delta-xTB models (d1/mace_torch_*_dxtb.yaml) predict dft_dipole - xtb_dipole; the frame's xtb_dipole
# (GFN2-xTB, about the COM, d1/mlff_qa.py) is added to every prediction.
# Label dft_dipole = mu - q R_COM in e*A (d1/dft_seed_run.py); reported in Debye. MAE is over Cartesian components (as
# Mai et al.'s dipole MAE), |d mu| (norm of the error vector) is given too, plus the zero-model baseline (predict 0)
# and the per-config_type MAE pooled over the held-out splits. Translation check: one frame per file shifted by 3 A.
import argparse, collections, json, os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g16_io import MASS  # noqa: E402

DEBYE_EA = 0.2081943     # 1 D in e*A
SPLITS = ("validation", "final_test", "transfer_test", "train")


def centred(at):
    m = np.array([MASS[s] for s in at.get_chemical_symbols()])
    b = at.copy(); b.positions -= (m[:, None] * b.positions).sum(0) / m.sum()
    if "mult" in b.info:          # PolarMACE reads the spin multiplicity (2S+1) from info["spin"]
        b.info["spin"] = int(b.info["mult"])
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("out"); ap.add_argument("--models", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--model-type", default="DipoleMACE"); ap.add_argument("--add-xtb", action="store_true"); ap.add_argument("--train-max", type=int, default=600)
    a = ap.parse_args()
    from ase.io import read
    from mace.calculators import MACECalculator
    models = dict(x.split("=", 1) for x in a.models.split(","))
    res, frames = {}, {}
    for q, path in sorted(models.items()):
        calc = MACECalculator(model_paths=path, device=a.device, default_dtype="float32", model_type=a.model_type)
        for sp in SPLITS:
            f = f"{a.dataset}/{sp}_{q}.extxyz"
            if not os.path.exists(f):
                continue
            ats = read(f, ":" if sp != "train" else f":{a.train_max}")
            rows, shift = [], None
            for k, at in enumerate(ats):
                ref = np.asarray(at.info["dft_dipole"], dtype=float).ravel()
                c = centred(at); c.calc = calc; mdl = np.asarray(calc.get_property("dipole", c))
                pred = mdl + (np.asarray(at.info["xtb_dipole"], dtype=float).ravel() if a.add_xtb else 0.0)
                if k == 0:   # shift the raw frame, then centre: checks the full predictor (centring + model)
                    b = at.copy(); b.positions += [3.0, 0, 0]; b = centred(b); b.calc = calc
                    shift = float(np.abs(np.asarray(calc.get_property("dipole", b)) - mdl).max() / DEBYE_EA)
                rows.append(dict(pair=str(at.info.get("source", "")).rsplit("_q", 1)[0], ct=at.info.get("config_type", ""),
                                 n=len(at), ref=(ref / DEBYE_EA).tolist(), pred=(pred / DEBYE_EA).tolist()))
            R = np.array([r["ref"] for r in rows]); P = np.array([r["pred"] for r in rows]); E = P - R
            res[f"{q}|{sp}"] = dict(frames=len(rows), mae_D=float(np.abs(E).mean()), norm_err_D=float(np.linalg.norm(E, axis=1).mean()),
                                    rmse_D=float(np.sqrt((E ** 2).mean())), zero_mae_D=float(np.abs(R).mean()),
                                    mean_abs_mu_D=float(np.linalg.norm(R, axis=1).mean()), shift_3A_D=shift)
            frames[f"{q}|{sp}"] = rows
            r = res[f"{q}|{sp}"]
            print(f"{q} {sp:13s} n={r['frames']:5d}  MAE {r['mae_D']:.4f} D  |dmu| {r['norm_err_D']:.4f} D  RMSE {r['rmse_D']:.4f}"
                  f"  (zero-model MAE {r['zero_mae_D']:.4f}, <|mu|> {r['mean_abs_mu_D']:.3f})  3A shift {shift:.1e} D", flush=True)
    print("\n### dipole MAE per config_type, held-out splits pooled, D (G2: q0 <= 0.06, q1 <= 0.1; * = above)")
    for q in sorted(models):
        lim = 0.06 if q == "q0" else 0.1
        by = collections.defaultdict(list)
        for sp in SPLITS[:3]:
            for r in frames.get(f"{q}|{sp}", []):
                by[r["ct"]].append(np.abs(np.subtract(r["pred"], r["ref"])).mean())
        print(f"{q}: " + ", ".join(f"{ct} {np.mean(v):.3f}{'*' if np.mean(v) > lim else ''} ({len(v)})" for ct, v in sorted(by.items())))
    print("\n### worst held-out species (MAE, D)")
    for q in sorted(models):
        by = collections.defaultdict(list)
        for sp in SPLITS[:3]:
            for r in frames.get(f"{q}|{sp}", []):
                by[r["pair"]].append(np.abs(np.subtract(r["pred"], r["ref"])).mean())
        w = sorted(by, key=lambda p: -np.mean(by[p]))[:8]
        print(f"{q}: " + ", ".join(f"{p} {np.mean(by[p]):.3f}" for p in w))
    with open(a.out, "w") as f:
        json.dump(dict(models=models, summary=res, frames=frames), f)


if __name__ == "__main__":
    main()
