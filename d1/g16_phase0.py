# Phase 0 inputs for Gaussian on Hangang (docs/plan/charged_mlff_plan.md): what to check before producing data.
#   python g16_phase0.py <pilot dir: /var/tmp/$USER-pah/d1/dft_pilot> <mlff seed dir> <pahdb.xml> <outdir>
# Writes <outdir>/{timing,origin,pahdb}/*.gjf, list8.txt / list32.txt (by %nprocshared) and expected.json:
#  timing  the dft_pilot geometries (same 0.05 A displacement, seed 0) at 4-31G (energies must match PySCF b3lypg:
#          expected.json) and 6-31G* (cost); neutral and cation, C10-C144; 8 CPUs up to C54, 32 above.
#  origin  the naphthalene cation pilot geometry and the same shifted by +3 A in x: the reported dipoles differ by
#          q * 3 A in x if Gaussian's dipole is about the coordinate origin, by 0 if about the centre of nuclear charge.
#  pahdb   naphthalene neutral and cation at their PAHdb geometries, 6-31G* single point + Freq: energies and
#          frequencies against PAHdb v4.00 (total_e and transitions in expected.json).
import json, os, sys
from xml.etree.ElementTree import iterparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROUTE = "#P {u}B3LYP/{basis} {job} NoSymm SCF=(XQC,MaxConventionalCycles=64,MaxCycle=256)"
L = lambda e: e.tag.split("}")[-1]


def kid(e, t):
    for k in e:
        if L(k) == t:
            return k


def read_xyz(p):
    lines = open(p).read().split("\n"); n = int(lines[0])
    return [l.split()[0] for l in lines[2:2 + n]], np.array([[float(x) for x in l.split()[1:4]] for l in lines[2:2 + n]])


def gjf(path, sym, P, q, m, basis, job="Force", nproc=8, mem="16GB"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as g:
        g.write(f"%nprocshared={nproc}\n%mem={mem}\n" + ROUTE.format(u="U" if m > 1 else "", basis=basis, job=job)
                + f"\n\n{os.path.basename(path)[:-4]}\n\n{q} {m}\n")
        g.writelines(f"{s:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n" for s, (x, y, z) in zip(sym, P))
        g.write("\n")
    return path


def main():
    pilot, seed, xml, out = sys.argv[1:5]
    exp = {"timing_pyscf_b3lypg_4-31g": {}, "pahdb": {}}; files = []
    for name in ("naphthalene", "coronene", "ovalene", "circumcoronene", "C90H24_618", "C144H30_641"):
        for q in (0, 1):
            sym, P = read_xyz(f"{pilot}/xyz/{name}_q{q}.xyz")
            P = P + np.random.default_rng(0).normal(0, 0.05, P.shape)        # as dft_timing.py, seed 0
            big = sym.count("C") > 60
            for basis in ("4-31G", "6-31G*"):
                tag = basis.replace("*", "s").replace("-", "")
                files.append(gjf(f"{out}/timing/{name}_q{q}_{tag}.gjf", sym, P, q, 2 if q else 1, basis,
                                 nproc=32 if big else 8, mem="60GB" if big else "16GB"))
            r = f"{pilot}/out/{name}_q{q}_t8.json"
            if os.path.exists(r):
                j = json.load(open(r)); exp["timing_pyscf_b3lypg_4-31g"][f"{name}_q{q}"] = {
                    "energy_Ha": j["energy"], "dipole_debye_about_com": j["dipole_debye"], "pyscf_seconds_8_threads": j["t_total"]}
            if name == "naphthalene" and q == 1:
                files.append(gjf(f"{out}/origin/naph_q1_at0.gjf", sym, P, 1, 2, "4-31G"))
                files.append(gjf(f"{out}/origin/naph_q1_shift3x.gjf", sym, P + [3.0, 0, 0], 1, 2, "4-31G"))
    want = {"330": 0, "359": 1}                                              # naphthalene neutral, cation (PAHdb uids)
    for _, el in iterparse(open(xml, "rb"), events=("end",)):
        if L(el) != "specie":
            continue
        uid = el.get("uid")
        if uid in want:
            q = want[uid]
            atoms = [(int(kid(a, "type").text), [float(kid(a, c).text) for c in "xyz"]) for a in kid(el, "geometry")]
            sym = ["H" if z == 1 else "C" for z, _ in atoms]; P = np.array([p for _, p in atoms])
            files.append(gjf(f"{out}/pahdb/naph_uid{uid}_q{q}_freq.gjf", sym, P, q, 2 if q else 1, "6-31G*", job="Freq"))
            exp["pahdb"][f"uid{uid}"] = {"total_e_Ha": float(kid(el, "total_e").text), "method": kid(el, "method").text,
                                         "frequencies_cm-1_unscaled_as_stored": [float(kid(m, "frequency").text) for m in kid(el, "transitions")],
                                         "frequency_scale_attr": kid(kid(el, "transitions")[0], "frequency").get("scale")}
        el.clear()
    for n in (8, 32):                                   # one Slurm array per CPU count (--cpus-per-task)
        sel = [f for f in files if f"%nprocshared={n}\n" in open(f).read()]
        open(f"{out}/list{n}.txt", "w").write("\n".join(os.path.abspath(f) for f in sel) + "\n")
    json.dump(exp, open(f"{out}/expected.json", "w"), indent=1)
    print(f"{len(files)} inputs; lists {out}/list8.txt, list32.txt; expected values {out}/expected.json")


if __name__ == "__main__":
    main()
