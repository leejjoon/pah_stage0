# Single points on saved MD frames (aimd_pyscf.py --frames-fs), to test whether the persistent per-replica spread of
# the naphthalene cation's 1000-1300 cm^-1 IR strength at 300 K (docs/plan/cloud_aimd_handoff.md, "Extra cation
# replicas") comes from different electronic solutions (charge/spin localisation, symmetry breaking).
# Per frame: a cold-start SCF at exactly the MD settings (b3lypg, 6-31G* 6D, PySCF default grid, conv_tol 1e-9, no
# density carried over), then
#   - energy against the MD potential energy at the same step (qa Etot minus the kinetic energy of frames_vel; only
#     at QA steps), and the dipole against the MD dipole at the same step: a mismatch means the MD's warm-started SCF
#     was on another solution than a fresh guess finds;
#   - <S^2>, Mulliken charges and spin populations, and their asymmetry in the molecular frame (carbon principal
#     axes: long, short, normal): ring A minus ring B (long axis) and upper minus lower half (short axis);
#   - every --stab-every-th selected frame: UKS internal stability test, re-optimised until stable (at most 3 rounds,
#     as Gaussian's Stable=Opt), with the energy, dipole and asymmetry after it.
# One JSON line per frame, appended to --out; frames already in --out are skipped (restartable).
#   python frame_sp.py <run.npz> --out <run.sp.jsonl> [--every 5] [--stab-every 10] [--part 0 --nparts 4] [--threads 8]
#   [--frames 0,50]   (explicit frame indices, for timing)
import argparse, json, os, sys, time
import numpy as np

EA_PER_DEBYE = 1 / 4.80320471


def axes(pos_c):
    """carbon principal axes: rows long, short (in plane), normal; signs fixed by the first carbon"""
    x = pos_c - pos_c.mean(0)
    w, v = np.linalg.eigh(x.T @ x)          # ascending: normal, short, long
    ax = v[:, ::-1].T.copy()
    return ax


def asym(pop, pos, isC, ax, com):
    """population difference: long-axis halves (ring A - ring B) and short-axis halves, carbons only"""
    p = (pos - com) @ ax.T
    c = isC
    return [float(pop[c & (p[:, 0] > 0)].sum() - pop[c & (p[:, 0] < 0)].sum()),
            float(pop[c & (p[:, 1] > 0)].sum() - pop[c & (p[:, 1] < 0)].sum())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=5); ap.add_argument("--stab-every", type=int, default=10)
    ap.add_argument("--part", type=int, default=0); ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--frames")
    a = ap.parse_args()
    from pyscf import gto, dft, lib
    from ase.io import read
    from ase.units import Hartree, fs
    lib.num_threads(a.threads)
    z = np.load(a.npz)
    q, mult = int(z["charge"]), int(z["mult"])
    xyz = os.path.join(os.path.dirname(os.path.abspath(a.npz)), os.path.basename(str(z["xyz"])))
    at = read(xyz); sym = at.get_chemical_symbols(); mass = at.get_masses(); isC = np.array([s == "C" for s in sym])
    fp, fv, fr = z["frames_pos"], z["frames_vel"], int(z["frames_every"])
    n_nvt = int(round(float(z["nvt_ps"]) * 2000)); dip = z["dipole"]; qa = z["qa"]
    qa_e = {int(r[0]): float(r[2]) for r in qa}
    sel = [int(i) for i in a.frames.split(",")] if a.frames else list(range(0, len(fp), a.every))
    stab = set(sel[::a.stab_every]) if a.stab_every else set()
    sel = [k for j, k in enumerate(sel) if j % a.nparts == a.part]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["frame"] for l in open(a.out) if l.strip()}
    run = os.path.basename(a.npz)[:-4]
    for k in sel:
        if k in done:
            continue
        t0 = time.time()
        pos = fp[k]; step = n_nvt + (k + 1) * fr
        com = (mass[:, None] * pos).sum(0) / mass.sum(); ax = axes(pos[isC])
        mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", charge=q, spin=mult - 1, unit="Angstrom", cart=True,
                    verbose=0)
        mf = dft.UKS(mol); mf.xc = "b3lypg"; mf.conv_tol = 1e-9; mf.max_cycle = 200
        n = [0]; mf.callback = lambda env: n.__setitem__(0, n[0] + 1)
        e = mf.kernel(); newton = False
        if not mf.converged:
            mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ); newton = True

        def props(mf):
            mu = np.asarray(mf.dip_moment(unit="Debye", verbose=0)) * EA_PER_DEBYE - q * com
            (pa, pb), chg = mf.mulliken_pop(verbose=0)          # per AO (alpha, beta), per-atom charges
            sl = mf.mol.aoslice_by_atom()
            spin = np.array([(pa - pb)[i0:i1].sum() for _, _, i0, i1 in sl])
            return dict(e_eV=float(mf.e_tot) * Hartree, s2=float(mf.spin_square()[0]), dipole=mu.tolist(),
                        dipole_mol=(ax @ mu).tolist(), charge_asym=asym(chg, pos, isC, ax, com),
                        spin_asym=asym(spin, pos, isC, ax, com), charges=np.round(chg, 5).tolist(),
                        spins=np.round(spin, 5).tolist())

        r = dict(run=run, frame=k, step=step, converged=bool(mf.converged), newton=newton, cycles=n[0], **props(mf))
        mu_md = dip[step // 2 - n_nvt // 2 - 1]       # dipole index of NVE step s: (s - n_nvt)/2 - 1
        r["dipole_md"] = mu_md.tolist(); r["dipole_md_mol"] = (ax @ mu_md).tolist()
        r["d_dipole"] = float(np.linalg.norm(np.array(r["dipole"]) - mu_md))
        if step in qa_e:
            ke = 0.5 * (mass[:, None] * fv[k] ** 2).sum()     # ASE units: eV
            r["e_md_pot_eV"] = qa_e[step] - ke; r["d_e_meV"] = (r["e_eV"] - r["e_md_pot_eV"]) * 1000
        r["t_scf"] = round(time.time() - t0, 1)
        if k in stab:
            t1 = time.time(); nr = 0
            while True:
                mo, _, st, _ = mf.stability(internal=True, external=False, return_status=True)
                if st or nr == 3:
                    break
                nr += 1
                mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))
            r["stability"] = dict(stable=bool(st), rounds=nr, t=round(time.time() - t1, 1))
            if nr:
                r["stability"].update(props(mf))
                r["stability"]["d_e_meV"] = (r["stability"]["e_eV"] - r["e_eV"]) * 1000
        r["host"] = os.uname().nodename
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{run} frame {k}: {r['t_scf']} s, {r['cycles']} cycles, |d mu| {r['d_dipole']:.2e}"
              + (f", dE {r['d_e_meV']:+.3f} meV" if "d_e_meV" in r else "")
              + (f", stable {r['stability']['stable']} ({r['stability']['t']} s)" if "stability" in r else ""),
              flush=True)


if __name__ == "__main__":
    main()
