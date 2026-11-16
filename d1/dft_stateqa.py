# Electronic-state QA side set (docs/plan/charged_mlff_plan.md, "Quality checks", item 4), the GPU4PySCF counterpart
# of g16_stateqa.py: same jobs (the <frame>_qa_<variant>.gjf names of g16_stateqa.py make), same settings as the
# production labels (dft_seed_run.py: b3lypg, 6-31G* cart, (99, 590) grid, conv_tol 1e-9 Ha).
#   python dft_stateqa.py run <qa gjf dir> <label json dir> --out DIR [--names JOB,...] [--shard i/n]
#       one JSON per job; finished jobs are skipped. Variants:
#         mix     UKS from the default solution with the alpha HOMO/LUMO rotated by 45 deg (Gaussian Guess=Mix), then
#                 internal stability re-optimisation (<= 5 rounds): neutrals (is there a broken-symmetry solution below
#                 RKS?) and cations (another doublet state?)
#         huckel  cations: UKS from a Hueckel guess, then the same re-optimisation
#         roks    cations of 3 named pairs: ROKS (SCF on the GPU; GPU4PySCF has no ROKS gradient, so the gradient
#                 runs on the CPU with PySCF), force and dipole differences against the UKS label
#       dE = E(variant) - E(label); a variant lower by more than --tol-ha (1e-5) means the default SCF did not find
#       the lowest solution. Forces and dipole of a mix/huckel solution are computed only if |dE| > 1e-6 Ha.
#   python dft_stateqa.py report <out dir>
#       one line per job, the frames with dE < -tol, and roks force/dipole differences.
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g16_io import DEBYE_EA, HB_TO_EVA, MASS


def build(sym, pos, q, m, kind):
    from pyscf import gto
    from gpu4pyscf import dft
    mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", cart=True, charge=q, spin=m - 1, unit="Angstrom", verbose=0)
    mf = getattr(dft, kind)(mol, xc="b3lypg")
    mf.grids.atom_grid = (99, 590); mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200
    return mf


def host(x):
    return np.asarray(x.get() if hasattr(x, "get") else x)


def converge(mf, **kw):
    e = mf.kernel(**kw); newton = False
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ); newton = True
    return mf, float(e), newton


def reopt(mf, e):
    n = 0
    while True:
        mo, _, st, _ = mf.stability(internal=True, external=False, return_status=True)
        if st or n == 5:
            return mf, e, n, bool(st)
        n += 1
        e = float(mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ)))


def mixed_guess(mf_ref, nalpha):
    """(2, nao, nmo) UKS orbitals from a reference solution with the alpha HOMO/LUMO rotated by 45 deg."""
    mo = host(mf_ref.mo_coeff)
    mo = np.stack([mo, mo]) if mo.ndim == 2 else mo.copy()
    h, l = mo[0][:, nalpha - 1].copy(), mo[0][:, nalpha].copy()
    mo[0][:, nalpha - 1], mo[0][:, nalpha] = (h + l) / np.sqrt(2), (h - l) / np.sqrt(2)
    return mo


def props(mf, sym, pos, q, cpu=False):
    if cpu:
        mf = mf.to_cpu()
    g = host(mf.nuc_grad_method().kernel())
    mu = host(mf.dip_moment(unit="Debye", verbose=0))
    m = np.array([MASS[s] for s in sym]); com = (m[:, None] * np.asarray(pos)).sum(0) / m.sum()
    return -g * HB_TO_EVA, mu * DEBYE_EA - q * com


def run_job(job, lab, tol):
    sym, pos = lab["symbols"], np.asarray(lab["positions"])
    q, m = int(lab["labels"]["charge"]), int(lab["labels"]["mult"])
    variant = job.rsplit("_qa_", 1)[1]
    t0 = time.time(); out = dict(job=job, frame=lab["name"], variant=variant, charge=q, mult=m, nC=sym.count("C"),
                                 config_type=lab["labels"].get("config_type"), e_label_Ha=lab["energy_Ha"],
                                 s2_label=lab["s2"])
    if variant == "roks":
        mf, e, newton = converge(build(sym, pos, q, m, "ROKS"))
        f, mu = props(mf, sym, pos, q, cpu=True)
        out.update(energy_Ha=e, newton=newton, s2=float("nan"), reopt=0, stable=None, have_props=True)
    else:
        if variant == "huckel":
            mf = build(sym, pos, q, m, "UKS"); mf.init_guess = "huckel"; mf, e, newton = converge(mf)
        else:
            ref, _, _ = converge(build(sym, pos, q, m, "RKS" if m == 1 else "UKS"))
            mf = build(sym, pos, q, m, "UKS")
            mo = mixed_guess(ref, ref.mol.nelec[0])
            occ = np.zeros((2, mo.shape[2])); occ[0, :ref.mol.nelec[0]] = 1; occ[1, :ref.mol.nelec[1]] = 1
            import cupy
            mf, e, newton = converge(mf, dm0=mf.make_rdm1(cupy.asarray(mo), cupy.asarray(occ)))
        mf, e, n, st = reopt(mf, e)
        out.update(energy_Ha=e, newton=newton, s2=float(mf.spin_square()[0]), reopt=n, stable=st,
                   have_props=abs(e - lab["energy_Ha"]) > 1e-6)
        if out["have_props"]:
            f, mu = props(mf, sym, pos, q)
    out["dE_Ha"] = out["energy_Ha"] - lab["energy_Ha"]
    out["lower"] = out["dE_Ha"] < -tol
    if out["have_props"]:
        df = f - np.asarray(lab["forces_eVA"])
        out.update(dF_max_eVA=float(np.abs(df).max()), dF_rms_eVA=float(np.sqrt((df ** 2).mean())),
                   dmu_eA=float(np.linalg.norm(mu - np.asarray(lab["dipole_eA"]))), forces_eVA=f.tolist(),
                   dipole_eA=mu.tolist())
    out["seconds"] = round(time.time() - t0, 1)
    return out


def cmd_run(a):
    jobs = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{a.qadir}/*_qa_*.gjf"))
    if a.names:
        keep = set(a.names.split(",")); jobs = [j for j in jobs if j in keep]
    i, n = map(int, a.shard.split("/")); jobs = jobs[i::n]
    os.makedirs(a.out, exist_ok=True)
    todo = [j for j in jobs if not os.path.exists(f"{a.out}/{j}.json")]
    print(f"shard {a.shard}: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    for k, j in enumerate(todo):
        lab = json.load(open(f"{a.labels}/{j.rsplit('_qa_', 1)[0]}.json"))
        try:
            r = run_job(j, lab, a.tol_ha)
        except Exception as ex:                       # report, do not stop the batch
            r = dict(job=j, error=f"{type(ex).__name__}: {ex}")
        json.dump(r, open(f"{a.out}/.{j}.tmp", "w")); os.replace(f"{a.out}/.{j}.tmp", f"{a.out}/{j}.json")
        print(f"[{k + 1}/{len(todo)}] {j} " + (r["error"] if "error" in r else
              f"dE={r['dE_Ha']:+.2e} Ha s2={r['s2']:.4f} reopt={r['reopt']}" +
              (f" dFmax={r['dF_max_eVA']:.3f} eV/A dmu={r['dmu_eA']:.3f} eA" if r["have_props"] else "") +
              f" {'LOWER' if r['lower'] else ''} t={r['seconds']}s"), flush=True)


def cmd_report(a):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/*_qa_*.json"))]
    err = [r for r in R if "error" in r]; R = [r for r in R if "error" not in r]
    print(f"{len(R)} jobs, {len(err)} errors {[r['job'] for r in err]}")
    for v in ("mix", "huckel", "roks"):
        rs = [r for r in R if r["variant"] == v]
        if not rs:
            continue
        de = np.array([r["dE_Ha"] for r in rs])
        print(f"{v}: {len(rs)} jobs; dE min {de.min():+.2e} max {de.max():+.2e} Ha; lower than the label (> tol): "
              f"{[(r['frame'], round(r['dE_Ha'], 6), round(r['s2'], 3)) for r in rs if r['lower']]}")
        if v == "roks":
            for r in rs:
                print(f"  {r['frame']}: dE {r['dE_Ha']:+.5f} Ha  dF max {r['dF_max_eVA']:.3f} rms {r['dF_rms_eVA']:.4f} eV/A"
                      f"  dmu {r['dmu_eA']:.4f} e*A")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("qadir"); r.add_argument("labels"); r.add_argument("--out", required=True)
    r.add_argument("--names", default=None); r.add_argument("--shard", default="0/1")
    r.add_argument("--tol-ha", type=float, default=1e-5)
    p = sub.add_parser("report"); p.add_argument("out")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
