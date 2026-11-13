# Foundation-model timing test, AIMNet2 (committee review M4): "aimnet2" (closed-shell, wB97M-D3) on neutrals and
# "aimnet2-nse" (open-shell, spin-resolved charges) on neutrals and cations, one GPU. Same species and measurements as
# d1/fm_timing.py (MACE-OMOL). Run with the aimnet venv built on container B's Python:
#   apptainer exec --nv B_torch.sif /var/tmp/$USER-pah/venvs/aimnet/bin/python d1/fm_timing_aimnet.py <xyz dir>
# Per model, species and charge: seconds per MD step for one molecule (ASE velocity Verlet, 0.5 fs), output keys, the
# dipole from the predicted atomic charges about the centre of mass, peak GPU memory, and batched throughput (B
# displaced geometries of one molecule as one (B, N) call) for B = 1-64.
import argparse, os, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz_dir"); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--species", default="C10H8_330,C24H12_18,C54H18_37,C96H26_9397")
    a = ap.parse_args()
    from ase import units
    from ase.io import read
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from aimnet.calculators import AIMNet2Calculator
    from aimnet.calculators.aimnet2ase import AIMNet2ASE
    print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
    for model, cases in (("aimnet2", ((0, 1),)), ("aimnet2-nse", ((0, 1), (1, 2)))):
        t0 = time.time(); base = AIMNet2Calculator(model, device="cuda")
        print(f"\n== {model}: loaded in {time.time() - t0:.1f} s; open-shell {base.is_nse}")
        for sp in a.species.split(","):
            for q, m in cases:
                f = f"{a.xyz_dir}/{sp}_q{q}.xyz"
                if not os.path.exists(f):
                    continue
                at = read(f)
                # The PAHdb xyz comment ("... charge 0 ...") makes ASE set info["charge"] = True, and AIMNet2ASE reads
                # atoms.info before its own arguments: set charge and multiplicity explicitly.
                at.info = {"charge": q, "mult": m, "spin": m}
                at.calc = AIMNet2ASE(base, charge=q, mult=m)
                torch.cuda.reset_peak_memory_stats()
                e = at.get_potential_energy(); keys = sorted(at.calc.results.keys())
                ch = at.calc.results.get("charges")
                dip = ""
                if ch is not None:
                    mass = at.get_masses(); com = (mass[:, None] * at.positions).sum(0) / mass.sum()
                    mu = (np.asarray(ch)[:, None] * (at.positions - com)).sum(0)
                    dip = f"; sum(q) {float(np.sum(ch)):+.3f}, |mu| {np.linalg.norm(mu) * 4.80320471:.3f} D"
                MaxwellBoltzmannDistribution(at, temperature_K=300, rng=np.random.default_rng(0))
                md = VelocityVerlet(at, 0.5 * units.fs); md.run(5)
                torch.cuda.synchronize(); t = time.time(); md.run(a.steps); torch.cuda.synchronize()
                ms = (time.time() - t) / a.steps * 1e3; mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"   {sp:12s} q={q} m={m} {len(at):3d} atoms: {ms:7.2f} ms per step ({ms * 2:.0f} s per ps), "
                      f"peak {mem:.2f} GB; outputs {keys}{dip}")
                rng = np.random.default_rng(1); line = []
                for nb in (1, 8, 32, 64):
                    coord = torch.tensor(np.stack([at.positions + rng.normal(0, 0.02, at.positions.shape) for _ in range(nb)]),
                                         dtype=torch.float32, device="cuda")
                    data = {"coord": coord, "numbers": torch.tensor(np.tile(at.numbers, (nb, 1)), device="cuda"),
                            "charge": torch.full((nb,), float(q), device="cuda"), "mult": torch.full((nb,), float(m), device="cuda")}
                    try:
                        base(data, forces=True); torch.cuda.synchronize(); t = time.time()
                        for _ in range(5):
                            out = base(data, forces=True)
                        torch.cuda.synchronize(); dt = (time.time() - t) / 5
                        line.append(f"{nb}: {dt / nb * 1e3:.2f} ms/mol")
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
                        if "out of memory" not in str(err):
                            raise
                        line.append(f"{nb}: OOM"); break
                    finally:
                        torch.cuda.empty_cache()
                print("      batched (energy+forces per molecule): " + "; ".join(line))
        del base; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
