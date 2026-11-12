# Gaussian 16 input writer and output parser for charged-state MLFF training data (docs/plan/charged_mlff_plan.md).
#   python g16_io.py write <frames.extxyz> <gjf dir> [--basis 6-31G*] [--nproc 8] [--mem 16GB] [--tag NAME]
#       one .gjf per frame: (U)B3LYP/<basis> Force NoSymm, SCF=(XQC,MaxConventionalCycles=64,MaxCycle=256) as PAHdb;
#       charge and multiplicity from the frame; frame info goes to <gjf dir>/manifest.json, keyed by job name.
#   python g16_io.py parse <log dir or files...> <out.extxyz> [--origin coords|nuclear_charge]
#       energy (eV), forces (eV/A), dipole about the centre of mass (e*A), <S^2> before annihilation, SCF cycles;
#       only "Normal termination" logs; failures are listed on stderr.
# Gaussian's B3LYP uses VWN3 (PySCF "b3lypg"). With NoSymm the coordinates and forces stay in the input frame.
# Dipole origin: --origin says which point Gaussian's reported dipole refers to; the Phase 0 translation test
# (docs/plan/charged_mlff_plan.md) decides it. For neutrals it does not matter.
import argparse, glob, json, os, re, sys
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
    for k, (info, sym, p) in enumerate(read_extxyz(a.extxyz)):
        q, m = int(info["charge"]), int(info["mult"])
        name = f"{tag}_f{k:03d}"; title = name
        manifest[name] = {**{k_: v.strip('"') for k_, v in info.items() if k_ != "Properties"}, "frame": k, "basis": a.basis}
        with open(f"{a.outdir}/{name}.gjf", "w") as g:
            g.write(f"%nprocshared={a.nproc}\n%mem={a.mem}\n"
                    f"#P {'U' if m > 1 else ''}B3LYP/{a.basis} Force NoSymm SCF=(XQC,MaxConventionalCycles=64,MaxCycle=256)\n\n"
                    f"{title}\n\n{q} {m}\n")
            g.writelines(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n" for s, (x, y, z) in zip(sym, p))
            g.write("\n")
    json.dump(manifest, open(mpath, "w"), indent=0)
    print(f"{a.extxyz}: {k + 1} inputs in {a.outdir}")


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
    s2 = re.findall(r"S\*\*2 before annihilation\s+(\d+\.\d+)", txt)
    mass = np.array([MASS[s] for s in sym]); com = (mass[:, None] * P).sum(0) / mass.sum()
    zn = np.array(Z, float); cnc = (zn[:, None] * P).sum(0) / zn.sum()
    o = np.zeros(3) if origin == "coords" else cnc
    mu_com = mu - q * (com - o)               # mu about point B = mu about A - q (B - A)
    return dict(sym=sym, P=P, F=F, energy=float(scf[1]) * HARTREE_EV, dipole=mu_com, charge=q, mult=mult,
                s2=float(s2[-1]) if s2 else 0.0, scf_cycles=int(scf[2]), method=scf[0]), None


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
                    "charge": r["charge"], "mult": r["mult"], "s2": f"{r['s2']:.4f}", "scf_cycles": r["scf_cycles"],
                    "method": r["method"], "basis": t.get("basis", "?"), "config_type": t.get("config_type", "?"),
                    "sample_T": t.get("sample_T", "?"), "source": t.get("source", "?"), "log": os.path.basename(path),
                    "pbc": '"F F F"'}
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
    p = sub.add_parser("parse"); p.add_argument("inputs", nargs="+"); p.add_argument("out")
    p.add_argument("--manifest", default=None, help="manifest.json from write (default: next to each log)")
    p.add_argument("--origin", choices=["coords", "nuclear_charge"], default="coords")
    a = ap.parse_args()
    (cmd_write if a.cmd == "write" else cmd_parse)(a)


if __name__ == "__main__":
    main()
