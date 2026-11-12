# Gaussian 16 input writer and output parser for charged-state MLFF training data (docs/plan/charged_mlff_plan.md).
#   python g16_io.py write <frames.extxyz> <gjf dir> [--basis 6-31G*] [--nproc 8] [--mem 16GB] [--tag NAME]
#        [--stable-hot-T 1200] [--stable-frac 0.1] [--no-stable]
#       one .gjf per frame: (U)B3LYP/<basis> Force NoSymm Int=UltraFine, SCF=(XQC,MaxConventionalCycles=64,
#       MaxCycle=256) as PAHdb; charge and multiplicity from the frame; frame info (all sampler labels) goes to
#       <gjf dir>/manifest.json, keyed by job name. Frames whose bond graph changed (graph_changed=1), controlled
#       boundary geometries (config_type bound_*) and a random --stable-frac of frames at >= --stable-hot-T get a
#       two-step job instead (committee review B2/M2): Stable=Opt (wavefunction stability test, re-optimised if
#       unstable), then --Link1-- Force from that wavefunction (Geom=AllCheck Guess=Read; %chk next to the input).
#   python g16_io.py parse <log dir or files...> <out.extxyz> [--origin coords|nuclear_charge]
#       energy (eV), forces (eV/A), dipole about the centre of mass (e*A), <S^2> before and after annihilation,
#       SCF cycles, stability result (stable / reoptimized / not_tested), plus every label from the manifest;
#       only "Normal termination" logs; failures are listed on stderr.
# Gaussian's B3LYP uses VWN3 (PySCF "b3lypg"). With NoSymm the coordinates and forces stay in the input frame.
# Dipole origin: --origin says which point Gaussian's reported dipole refers to; the Phase 0 translation test
# (docs/plan/charged_mlff_plan.md) decides it. For neutrals it does not matter.
import argparse, glob, hashlib, json, os, re, sys
import numpy as np

HARTREE_EV = 27.211386245988
HB_TO_EVA = 51.422086190832
DEBYE_EA = 1 / 4.80320471
MASS = {"H": 1.00782503, "C": 12.0, "N": 14.00307401, "O": 15.99491462}
ZNUC = {"H": 1, "C": 6, "N": 7, "O": 8}
SYM = {v: k for k, v in ZNUC.items()}


def read_extxyz(path):
    frames = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line.strip():
                break
            n = int(line); info = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', f.readline()))
            rows = [f.readline().split() for _ in range(n)]
            frames.append((info, [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])))
    return frames


def cmd_write(a):
    os.makedirs(a.outdir, exist_ok=True)
    tag = a.tag or os.path.basename(a.extxyz).rsplit(".", 1)[0]
    mpath = f"{a.outdir}/manifest.json"
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    n_stable = 0
    for k, (info, sym, p) in enumerate(read_extxyz(a.extxyz)):
        q, m = int(info["charge"]), int(info["mult"])
        name = f"{tag}_f{k:03d}"; title = name
        lab = {k_: v.strip('"') for k_, v in info.items() if k_ != "Properties"}
        T = float(lab["sample_T"]) if lab.get("sample_T", "none").replace(".", "").isdigit() else 0.0
        u = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 16**8      # reproducible per-frame random number
        stable = not a.no_stable and (lab.get("graph_changed") == "1" or lab.get("config_type", "").startswith("bound")
                                      or (T >= a.stable_hot_T and u < a.stable_frac))
        n_stable += stable
        manifest[name] = {**lab, "frame": k, "basis": a.basis, "stability_test": int(stable)}
        u_ = "U" if m > 1 else ""
        scf = "SCF=(XQC,MaxConventionalCycles=64,MaxCycle=256)"
        head = f"%nprocshared={a.nproc}\n%mem={a.mem}\n" + (f"%chk={name}.chk\n" if stable else "")
        with open(f"{a.outdir}/{name}.gjf", "w") as g:
            g.write(head + f"#P {u_}B3LYP/{a.basis} {'Stable=Opt' if stable else 'Force'} NoSymm Int=UltraFine {scf}\n\n"
                    f"{title}\n\n{q} {m}\n")
            g.writelines(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n" for s, (x, y, z) in zip(sym, p))
            g.write("\n")
            if stable:
                g.write(f"--Link1--\n{head}#P {u_}B3LYP/{a.basis} Force NoSymm Int=UltraFine {scf} Geom=AllCheck Guess=Read\n\n")
    json.dump(manifest, open(mpath, "w"), indent=0)
    print(f"{a.extxyz}: {k + 1} inputs in {a.outdir} ({n_stable} with a stability test)")


def parse_log(path, origin="coords"):
    txt = open(path, errors="replace").read()
    if "Normal termination" not in txt[-3000:]:
        return None, "no normal termination"
    blocks = re.findall(r"Input orientation:.*?-{20,}\n.*?-{20,}\n(.*?)\n -{20,}", txt, re.S)
    rows = [l.split() for l in blocks[-1].splitlines()]
    Z = [int(r[1]) for r in rows]; P = np.array([[float(x) for x in r[3:6]] for r in rows])
    sym = [SYM[z] for z in Z]
    scf = re.findall(r"SCF Done:\s+E\((\w+)\)\s+=\s+(-?\d+\.\d+)\s+A\.U\. after\s+(\d+) cycles", txt)[-1]
    fb = re.search(r"Forces \(Hartrees/Bohr\)\n.*?\n -{20,}\n(.*?)\n -{20,}", txt, re.S).group(1)
    F = np.array([[float(x) for x in l.split()[2:5]] for l in fb.splitlines()]) * HB_TO_EVA
    d = re.findall(r"Dipole moment \(field-independent basis, Debye\):\s*\n\s*X=\s*(\S+)\s+Y=\s*(\S+)\s+Z=\s*(\S+)", txt)[-1]
    mu = np.array([float(x) for x in d]) * DEBYE_EA
    ch = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", txt)
    q, mult = int(ch.group(1)), int(ch.group(2))
    s2 = re.findall(r"S\*\*2 before annihilation\s+(\d+\.\d+),\s+after\s+(\d+\.\d+)", txt)
    stab = ("reoptimized" if re.search(r"wavefunction has an? .*instability", txt) else "stable") \
        if "Stability analysis" in txt or "wavefunction is stable" in txt else "not_tested"
    mass = np.array([MASS[s] for s in sym]); com = (mass[:, None] * P).sum(0) / mass.sum()
    zn = np.array(Z, float); cnc = (zn[:, None] * P).sum(0) / zn.sum()
    o = np.zeros(3) if origin == "coords" else cnc
    mu_com = mu - q * (com - o)               # mu about point B = mu about A - q (B - A)
    return dict(sym=sym, P=P, F=F, energy=float(scf[1]) * HARTREE_EV, dipole=mu_com, charge=q, mult=mult,
                s2=float(s2[-1][0]) if s2 else 0.0, s2_after=float(s2[-1][1]) if s2 else 0.0, stability=stab,
                scf_cycles=int(scf[2]), method=scf[0]), None


def cmd_parse(a):
    files = []
    for x in a.inputs:
        files += sorted(glob.glob(f"{x}/*.log")) if os.path.isdir(x) else [x]
    ok = 0; manifests = {}
    with open(a.out, "w") as f:
        for path in files:
            try:
                r, err = parse_log(path, a.origin)
            except Exception as e:                       # malformed or truncated output
                r, err = None, f"parse error: {e}"
            if r is None:
                print(f"{path}: {err}", file=sys.stderr); continue
            d = os.path.dirname(os.path.abspath(path))
            if d not in manifests:                       # frame info written by "write" (manifest.json next to the logs)
                mp = a.manifest or f"{d}/manifest.json"
                manifests[d] = json.load(open(mp)) if os.path.exists(mp) else {}
            t = manifests[d].get(os.path.basename(path).rsplit(".", 1)[0], {})
            info = {"dft_energy": f"{r['energy']:.8f}", "dft_dipole": '"' + " ".join(f"{x:.6f}" for x in r["dipole"]) + '"',
                    "charge": r["charge"], "mult": r["mult"], "s2": f"{r['s2']:.4f}", "s2_after": f"{r['s2_after']:.4f}",
                    "stability": r["stability"], "scf_cycles": r["scf_cycles"], "method": r["method"],
                    **{k: v for k, v in t.items() if k not in ("charge", "mult", "frame")},
                    "log": os.path.basename(path), "pbc": '"F F F"'}
            f.write(f"{len(r['sym'])}\nProperties=species:S:1:pos:R:3:dft_forces:R:3 "
                    + " ".join(f"{k}={v}" for k, v in info.items()) + "\n")
            f.writelines(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {g[0]:.8f} {g[1]:.8f} {g[2]:.8f}\n"
                         for s, p, g in zip(r["sym"], r["P"], r["F"]))
            ok += 1
    print(f"{ok}/{len(files)} logs parsed -> {a.out}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.add_argument("extxyz"); w.add_argument("outdir")
    w.add_argument("--basis", default="6-31G*"); w.add_argument("--nproc", type=int, default=8)
    w.add_argument("--mem", default="16GB"); w.add_argument("--tag", default=None)
    w.add_argument("--stable-hot-T", type=float, default=1200.0); w.add_argument("--stable-frac", type=float, default=0.1)
    w.add_argument("--no-stable", action="store_true", help="never write the two-step stability job")
    p = sub.add_parser("parse"); p.add_argument("inputs", nargs="+"); p.add_argument("out")
    p.add_argument("--manifest", default=None, help="manifest.json from write (default: next to each log)")
    p.add_argument("--origin", choices=["coords", "nuclear_charge"], default="coords")
    a = ap.parse_args()
    (cmd_write if a.cmd == "write" else cmd_parse)(a)


if __name__ == "__main__":
    main()
