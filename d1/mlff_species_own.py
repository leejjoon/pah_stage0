# AL round 3, track 2 (docs/plan/charged_mlff_plan.md): hand-built species outside PAHdb, with the motifs behind v3's
# cation error (five-membered rings, CH2/CH3 sp3 carbons, biaryls), user approval 2026-11-21 (coordinator note).
#   python mlff_species_own.py build <smiles.csv> <pahdb.xml> <species csv,...> <out dir>     (env mamba/rdkit)
# <smiles.csv>: columns name, smiles, motif. Per row: RDKit (ETKDGv3, seed 0) + MMFF, then GFN2-xTB optimisation
# (tblite, ASE BFGS, fmax 0.02 eV/A) of the neutral. Then, as for every other species, the carbon skeleton
# (mlff_splits.skeleton) is checked against every species table given: a skeleton in a held-out split is rejected
# (this also rejects every hydrogenated or substituted-at-H form of a held-out skeleton, since H atoms are not in the
# skeleton); a skeleton in train is accepted (new H pattern on a train skeleton); an unused skeleton is accepted and
# claimed. The full element graph is checked against PAHdb v4.00 (a molecule PAHdb has is not hand-built) and
# against the other rows (duplicates). Accepted rows all go to train. Writes <out>/species_own.csv (species_v2
# columns + feature columns as species_r2, smiles, motif, species_own=1, status of every row in <out>/build_log.csv)
# and <out>/xyz_start/<pair>_q0.xyz and _q1.xyz (the xTB geometry; B3LYP minima of both charges are optimised next
# with `mlff_species_r2.py opt --charge 0|1`). Pair names: <formula>_x<NN>, NN the row number (no PAHdb uid).
import argparse, csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_one(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ase import Atoms
    from ase.optimize import BFGS
    from tblite.ase import TBLite
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3(); p.randomSeed = 0
    if AllChem.EmbedMolecule(m, p) != 0:
        raise RuntimeError("embedding failed")
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    at = Atoms([a.GetSymbol() for a in m.GetAtoms()], positions=m.GetConformer().GetPositions())
    at.calc = TBLite(method="GFN2-xTB", verbosity=0)
    BFGS(at, logfile=None).run(fmax=0.02, steps=1000)
    Z = at.get_atomic_numbers(); P = at.get_positions() - at.get_center_of_mass()
    return Z, P, float(at.get_potential_energy())


def edges(Z, P):
    """solo/duo/trio/quartet counts: runs of adjacent CH carbons on aromatic (3-connected) rings, as PAHdb bec()."""
    n = len(Z); d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    A = (d < np.where((Z[:, None] == 6) & (Z[None] == 6), 1.75, 1.25)) & ~np.eye(n, dtype=bool)
    ch = {i for i in range(n) if Z[i] == 6 and A[i].sum() == 3 and any(Z[j] == 1 for j in np.nonzero(A[i])[0])}
    seen, out = set(), {"solo": 0, "duo": 0, "trio": 0, "quartet": 0}
    for s in ch:
        if s in seen:
            continue
        comp, st = set(), [s]
        while st:
            i = st.pop()
            if i in comp:
                continue
            comp.add(i); st += [j for j in np.nonzero(A[i])[0] if j in ch]
        seen |= comp
        k = {1: "solo", 2: "duo", 3: "trio", 4: "quartet"}.get(len(comp))
        if k:
            out[k] += 1
    return out


def cmd_build(a):
    from mlff_species import graph_hash, read_pahdb
    from mlff_splits import FIELDS, skeleton
    from mlff_species_r2 import features, write_xyz
    split_of = {}
    for f in a.species.split(","):
        for r in csv.DictReader(open(f)):
            split_of.setdefault(r["skeleton"], set()).add(r["split"])
    pahdb = {graph_hash(s["Z"], s["P"]) for s in read_pahdb(a.xml)}
    rows = list(csv.DictReader(open(a.smiles)))
    os.makedirs(f"{a.out}/xyz_start", exist_ok=True)
    extra = ["n_ring", "n_ring5", "n_ring7", "n_sp3", "oop_rms"]
    acc, log, seen_full, claimed = [], [], {}, {}
    for k, r in enumerate(rows, 1):
        st, Z, P = "", None, None
        try:
            Z, P, _ = build_one(r["smiles"])
        except Exception as e:  # noqa: BLE001
            st = f"build failed: {e}"
        if not st:
            sk, full = skeleton(Z, P), graph_hash(Z, P)
            held = split_of.get(sk, set()) - {"train"}
            if held:
                st = f"rejected: skeleton {sk} held out ({','.join(sorted(held))})"
            elif full in pahdb:
                st = "rejected: molecule is in PAHdb"
            elif full in seen_full:
                st = f"rejected: duplicate of {seen_full[full]}"
            else:
                nC = int((Z == 6).sum()); nH = int((Z == 1).sum()); formula = f"C{nC}H{nH}"; pair = f"{formula}_x{k:02d}"
                seen_full[full] = r["name"]
                st = "accepted: " + ("train skeleton" if sk in split_of else
                                     f"new skeleton{' (shared with ' + claimed[sk] + ')' if sk in claimed else ''}")
                claimed.setdefault(sk, pair)
                e = edges(Z, P); ft = features(Z, P)
                acc.append(dict(zip(FIELDS, [pair, formula, nC, "train", sk, "", "", "", "", "", "6-31g*(own)", "6-31g*(own)",
                                             e["solo"], e["duo"], e["trio"], e["quartet"]]))
                           | {x: ft[x] for x in extra} | dict(smiles=r["smiles"], motif=r["motif"], name=r["name"], species_own=1))
                for q in (0, 1):
                    write_xyz(f"{a.out}/xyz_start/{pair}_q{q}.xyz", Z, P,
                              f"own species {r['name']} ({r['smiles']}), GFN2-xTB neutral geometry; start for B3LYP q{q}")
        log.append(dict(row=k, name=r["name"], smiles=r["smiles"], motif=r["motif"], status=st))
        print(f"{k:3d} {r['name']:32s} {st}", flush=True)
    with open(f"{a.out}/build_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(log[0])); w.writeheader(); w.writerows(log)
    if acc:
        with open(f"{a.out}/species_own.csv", "w", newline="") as f:
            w = csv.DictWriter(f, list(acc[0])); w.writeheader(); w.writerows(acc)
    print(f"{len(acc)} of {len(rows)} accepted -> {a.out}/species_own.csv")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("smiles"); b.add_argument("xml"); b.add_argument("species"); b.add_argument("out")
    a = ap.parse_args()
    {"build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
