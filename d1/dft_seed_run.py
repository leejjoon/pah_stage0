# G4 of docs/plan/handoff_gpu4pyscf_gh200.md: B3LYP/6-31G* labels for the MLFF seed set with GPU4PySCF, the PySCF
# counterpart of the Gaussian route in g16_io.py (same frames, same frame names, same stability sample, same output).
#   python dft_seed_run.py run <frames.extxyz or dir>... --out DIR [--manifest gjf/manifest.json] [--shard i/n]
#                              [--only PAIR,...] [--names FRAME,...] [--limit N]
#       one JSON per frame, <out>/<pair>_q<q>_f<k>.json (the g16_io.py job name); finished frames are skipped, so a
#       killed run resumes. --shard i/n takes every n-th frame (largest molecules first) for one process per GPU.
#   python dft_seed_run.py collect <out dir> <parsed.extxyz>
#       all JSONs -> one extxyz in g16_io.py parse's format (dft_energy eV, dft_forces eV/A, dft_dipole e*A about the
#       centre of mass, s2, stability, scf_cycles, method, every manifest label), ready for mlff_qa.py; prints a QA
#       summary (unconverged, Newton fallbacks, <S^2>, instabilities, largest forces).
# Settings (docs/research/pyscf_gpu_jax.md §1.3, matched to Gaussian/PAHdb): b3lypg (VWN3), 6-31G* with 6D d
# (cart=True), (99, 590) grid, conv_tol 1e-9 Ha, conv_tol_grad 1e-6, max_cycle 200, then second-order SCF if needed.
# Stability (frames with stability_test=1 in the manifest, as g16_io.py write chose them): open shells Stable=Opt
# (internal UKS instability -> re-optimise, at most 5 rounds); closed shells test only (internal and RKS->UKS),
# recorded, not acted on, so the method stays RKS for every frame.
import argparse, glob, json, os, socket, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g16_io import DEBYE_EA, HARTREE_EV, HB_TO_EVA, MASS, read_extxyz


def frames_of(paths, manifest_path):
    """[(name, info, sym, pos, stability_test)] for every frame of the given extxyz files (or dirs of them)."""
    files = []
    for p in paths:
        files += sorted(glob.glob(f"{p}/*.extxyz")) if os.path.isdir(p) else [p]
    out = []
    for f in files:
        mp = manifest_path or os.path.join(os.path.dirname(os.path.abspath(f)), "..",
                                           "gjf" + os.path.basename(os.path.dirname(os.path.abspath(f)))[6:], "manifest.json")
        man = json.load(open(mp)) if os.path.exists(mp) else {}
        tag = os.path.basename(f).rsplit(".", 1)[0]
        for k, (info, sym, pos) in enumerate(read_extxyz(f)):
            name = f"{tag}_f{k:03d}"
            lab = {a: v.strip('"') for a, v in info.items() if a != "Properties"}
            if name not in man:
                print(f"warning: {name} not in {mp}; no stability test", file=sys.stderr)
            out.append((name, lab, sym, pos, int(man.get(name, {}).get("stability_test", 0))))
    return out


def label(sym, pos, charge, mult, stab_test):
    import cupy
    from pyscf import gto
    from gpu4pyscf import dft
    mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", cart=True, charge=charge, spin=mult - 1, unit="Angstrom",
                verbose=0)
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol, xc="b3lypg")
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200      # GPU energy noise floor ~1e-9 Ha
    t0 = time.time()
    e = mf.kernel(); cycles = int(getattr(mf, "cycles", 0)); newton = False
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ); newton = True   # newton().kernel takes MOs first
    t_scf = time.time() - t0
    stability, reopt, e_pre = "not_tested", 0, None
    if stab_test:
        t1 = time.time()
        if mult == 1:     # Gaussian "Stable": test only
            _, _, st_i, st_e = mf.stability(internal=True, external=True, return_status=True)
            stability = "stable" if st_i and st_e else ("unstable_internal" if not st_i else "unstable_rhf_uhf")
        else:             # Gaussian "Stable=Opt"
            e_pre = float(e)
            while True:
                mo, _, st, _ = mf.stability(internal=True, external=False, return_status=True)
                if st or reopt == 5:
                    break
                reopt += 1
                e = mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
            stability = "stable" if reopt == 0 else ("reoptimized" if st else "unstable_internal")
        t_stab = time.time() - t1
    else:
        t_stab = 0.0
    t1 = time.time()
    g = mf.nuc_grad_method().kernel(); g = np.asarray(g.get() if hasattr(g, "get") else g)
    mu = mf.dip_moment(unit="Debye", verbose=0); mu = np.asarray(mu.get() if hasattr(mu, "get") else mu)
    t_grad = time.time() - t1
    m = np.array([MASS[s] for s in sym]); com = (m[:, None] * pos).sum(0) / m.sum()
    mu_com = mu * DEBYE_EA - charge * com            # e*A, about the centre of mass (mu_B = mu_A - q (B - A))
    s2 = float(mf.spin_square()[0]) if mult > 1 else 0.0
    return dict(energy_eV=float(e) * HARTREE_EV, energy_Ha=float(e), forces_eVA=(-g * HB_TO_EVA).tolist(),
                dipole_eA=mu_com.tolist(), s2=s2, converged=bool(mf.converged), newton=newton, scf_cycles=cycles,
                stability=stability, reopt_rounds=reopt, energy_before_reopt_Ha=e_pre,
                method=("R" if mult == 1 else "U") + "B3LYP", nao=int(mol.nao),
                t_scf=round(t_scf, 2), t_stab=round(t_stab, 2), t_grad=round(t_grad, 2),
                gpu_mem_gb=round(cupy.get_default_memory_pool().total_bytes() / 1e9, 3))


def cmd_run(a):
    import cupy, gpu4pyscf, pyscf
    fr = frames_of(a.frames, a.manifest)
    if a.only:
        keep = set(a.only.split(","))
        fr = [f for f in fr if f[0].rsplit("_q", 1)[0] in keep]
    fr.sort(key=lambda f: (-len(f[2]), f[0]))        # largest first, so the shards finish together
    i, n = map(int, a.shard.split("/"))
    fr = fr[i::n]
    if a.names:
        keep = set(a.names.split(","))
        fr = [f for f in fr if f[0] in keep]
    todo = [f for f in fr if not os.path.exists(f"{a.out}/{f[0]}.json")]
    n_done = len(fr) - len(todo)
    if a.limit:
        todo = todo[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    env = dict(gpu4pyscf=gpu4pyscf.__version__, pyscf=pyscf.__version__, cupy=cupy.__version__, host=socket.gethostname(),
               gpu=cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(f"shard {a.shard}: {len(fr)} frames, {n_done} done, {len(todo)} to run ({env['gpu']})", flush=True)
    t_all = time.time()
    for j, (name, lab, sym, pos, st) in enumerate(todo):
        t0 = time.time()
        r = label(sym, pos, int(lab["charge"]), int(lab["mult"]), st)
        r.update(name=name, labels=lab, stability_test=st, symbols=list(sym), positions=np.asarray(pos).tolist(),
                 t_wall=round(time.time() - t0, 2), date=time.strftime("%Y-%m-%d %H:%M:%S"), **env)
        tmp = f"{a.out}/.{name}.json.tmp"
        json.dump(r, open(tmp, "w")); os.replace(tmp, f"{a.out}/{name}.json")       # atomic: no half-written results
        print(f"[{j + 1}/{len(todo)}] {name} E={r['energy_Ha']:.9f} cyc={r['scf_cycles']}{'+N' if r['newton'] else ''} "
              f"conv={r['converged']} s2={r['s2']:.4f} stab={r['stability']} t={r['t_wall']}s "
              f"(elapsed {(time.time() - t_all) / 3600:.2f} h)", flush=True)


def cmd_collect(a):
    files = sorted(glob.glob(f"{a.outdir}/*.json"))
    R = [json.load(open(f)) for f in files]
    with open(a.out, "w") as f:
        for r in R:
            info = {"dft_energy": f"{r['energy_eV']:.8f}", "dft_dipole": '"' + " ".join(f"{x:.6f}" for x in r["dipole_eA"]) + '"',
                    "charge": r["labels"]["charge"], "mult": r["labels"]["mult"], "s2": f"{r['s2']:.4f}",
                    "stability": r["stability"], "scf_cycles": r["scf_cycles"], "method": r["method"],
                    "converged": int(r["converged"]), "newton": int(r["newton"]),
                    **{k: v for k, v in r["labels"].items() if k not in ("charge", "mult", "pbc")},
                    "log": r["name"], "engine": f'"gpu4pyscf {r["gpu4pyscf"]}"', "pbc": '"F F F"'}
            f.write(f"{len(r['symbols'])}\nProperties=species:S:1:pos:R:3:dft_forces:R:3 "
                    + " ".join(f"{k}={v}" for k, v in info.items()) + "\n")
            f.writelines(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {g[0]:.8f} {g[1]:.8f} {g[2]:.8f}\n"
                         for s, p, g in zip(r["symbols"], r["positions"], r["forces_eVA"]))
    print(f"{len(R)} frames -> {a.out}")
    bad = [r["name"] for r in R if not r["converged"]]
    newton = [r["name"] for r in R if r["newton"]]
    s2hi = [(r["name"], round(r["s2"], 3)) for r in R if int(r["labels"]["mult"]) == 2 and r["s2"] > 0.80]
    unst = [(r["name"], r["stability"]) for r in R if r["stability"] in ("unstable_internal", "unstable_rhf_uhf", "reoptimized")]
    fmax = sorted(((float(np.abs(r["forces_eVA"]).max()), r["name"], r["labels"].get("config_type")) for r in R), reverse=True)
    t = np.array([r["t_wall"] for r in R]) if R else np.zeros(1)
    print(f"unconverged: {len(bad)} {bad[:10]}\nnewton fallback: {len(newton)} {newton[:10]}\n"
          f"doublet <S^2> > 0.80: {len(s2hi)} {s2hi[:10]}\nstability (tested {sum(r['stability_test'] for r in R)}): "
          f"{len(unst)} not plainly stable {unst[:10]}\nlargest |F| component (eV/A): {[(round(x, 1), n, c) for x, n, c in fmax[:5]]}\n"
          f"GPU time: {t.sum() / 3600:.2f} h over {len(R)} frames (mean {t.mean():.1f} s)")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("frames", nargs="+"); r.add_argument("--out", required=True)
    r.add_argument("--manifest", default=None, help="default: ../gjf<suffix>/manifest.json next to each frames dir")
    r.add_argument("--shard", default="0/1"); r.add_argument("--only", default=None, help="comma-separated pairs")
    r.add_argument("--names", default=None, help="comma-separated frame names (e.g. C10H8_330_q1_f008)")
    r.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("collect"); c.add_argument("outdir"); c.add_argument("out")
    a = ap.parse_args()
    {"run": cmd_run, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
