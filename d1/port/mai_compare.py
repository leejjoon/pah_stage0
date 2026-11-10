# Step 2a: compare the port (container A, JAX 0.11) with the authors' pinned environment (mai_ref_eval.py).
#   python mai_compare.py prep    <xyz dir> <outdir>   write geoms/: the selected species at the PAHdb DFT geometry
#                                                      and one rattled copy each (sigma 0.05 A, fixed seed)
#   python mai_compare.py port    <Supplementary_Data_Code dir> <outdir>   port on GPU and CPU -> port_gpu/cpu.npz
#   python mai_compare.py compare <outdir>             per geometry: |dE|, max|dF|, max|dq|, |dmu| vs ref.npz
# Gates: forces and charges (energy is only reported; float32 spacing at -1e4..-2e5 eV is 1e-3..1e-2 eV).
import sys, os, glob
import numpy as np

SPECIES = ["C10H8_330",                    # naphthalene
           "C13H9N_274", "C12H8N2_267",    # N-containing: nonzero dipoles
           "C9H8_495",                     # indene: CH2, polar
           "C24H24_357", "C96H22_699", "C66H18_713",   # largest max|F| in the port sweep
           "C216H36_615"]                  # largest species (252 atoms)
TOL_F, TOL_Q, TOL_MU = 1e-3, 1e-4, 1e-3   # eV/A, e, e*A

def write_xyz(path, atoms):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n{os.path.basename(path)[:-4]}\n")
        for s, p in zip(atoms.get_chemical_symbols(), atoms.positions):
            f.write(f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")

mode = sys.argv[1]
if mode == "prep":
    from ase.io import read
    xyzdir, outdir = sys.argv[2], sys.argv[3]; os.makedirs(f"{outdir}/geoms", exist_ok=True)
    rng = np.random.default_rng(20261110)
    for n in SPECIES:
        a = read(f"{xyzdir}/{n}.xyz"); write_xyz(f"{outdir}/geoms/{n}.xyz", a)
        a.positions += rng.normal(0, 0.05, a.positions.shape); write_xyz(f"{outdir}/geoms/{n}_rattled.xyz", a)
    print(len(glob.glob(f"{outdir}/geoms/*.xyz")), "geometries in", f"{outdir}/geoms")

elif mode == "port":
    root, outdir = sys.argv[2], sys.argv[3]
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mai_port import Port
    m = Port(root)
    import jax
    from ase.io import read
    print("jax", jax.__version__, jax.devices(), jax.devices("cpu"))
    for dev in (jax.devices()[0], jax.devices("cpu")[0]):
        names, E, F, Q, MU = [], [], [], [], []
        with jax.default_device(dev):
            calc = m.calculator(n_devices=1)
            for f in sorted(glob.glob(f"{outdir}/geoms/*.xyz")):
                a = read(f); a.calc = calc
                E.append(a.get_potential_energy()); F.append(a.get_forces())
                mu, q = m.dipole(a); MU.append(mu); Q.append(q); names.append(os.path.basename(f)[:-4])
        np.savez(f"{outdir}/port_{dev.platform}.npz", names=names, E=np.array(E), **{f"F_{n}": x for n, x in zip(names, F)},
                 **{f"Q_{n}": x for n, x in zip(names, Q)}, MU=np.array(MU))
        print(dev, "done:", len(names), "geometries")

elif mode == "compare":
    outdir = sys.argv[2]; ref = np.load(f"{outdir}/ref.npz"); ok = True
    for tag in ("gpu", "cpu"):
        p = np.load(f"{outdir}/port_{tag}.npz")
        assert list(p["names"]) == list(ref["names"])
        print(f"port ({tag}) vs reference (JAX 0.4.19 CPU)")
        print(f"  {'geometry':24s} {'atoms':>5s} {'|dE| eV':>9s} {'max|dF|':>9s} {'max|F|':>7s} {'max|dq|':>9s} {'|mu|':>8s} {'|dmu|':>9s}")
        for i, n in enumerate(ref["names"]):
            dF = np.abs(p[f"F_{n}"] - ref[f"F_{n}"]).max(); dq = np.abs(p[f"Q_{n}"] - ref[f"Q_{n}"]).max()
            dmu = np.linalg.norm(p["MU"][i] - ref["MU"][i]); good = dF < TOL_F and dq < TOL_Q and dmu < TOL_MU; ok &= good
            print(f"  {n:24s} {len(ref[f'Q_{n}']):5d} {abs(p['E'][i] - ref['E'][i]):9.2e} {dF:9.2e} "
                  f"{np.linalg.norm(ref[f'F_{n}'], axis=1).max():7.3f} {dq:9.2e} {np.linalg.norm(ref['MU'][i]):8.4f} {dmu:9.2e}"
                  + ("" if good else "  FAIL"))
    print(f"gates: max|dF| < {TOL_F} eV/A, max|dq| < {TOL_Q} e, |dmu| < {TOL_MU} e*A ->", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
