# Foundation-model timing test (committee review M4; docs/plan/charged_mlff_plan.md): MACE-OMOL-0 (charge and spin
# embeddings) on PAH neutrals and cations, on one GPU. Run in container B (mace-torch 0.3.17):
#   apptainer exec --nv B_torch.sif python d1/fm_timing.py <model file> <xyz dir> [--steps 50]
# Per species, charge and dtype: seconds per MD step for one molecule (ASE velocity Verlet, 0.5 fs, energy + forces),
# the model's output keys (does it give charges or dipoles?), peak GPU memory, and batched throughput (several
# geometries of one molecule in one model call) for batch sizes 1-64. Charge and spin go in atoms.info["charge"] and
# atoms.info["spin"] (spin multiplicity, as in OMol25); the script also prints the doublet-minus-singlet energy for the
# cation as a sanity check of that convention.
import argparse, glob, os, time
import numpy as np
import torch


def load_xyz(path):
    from ase.io import read
    return read(path)


def batched_eval(calc, atoms_list):
    """One model call on a batch of molecules, built the way MACECalculator builds a single one."""
    from mace import data as mace_data
    from mace.tools import torch_geometric, torch_tools
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    with torch_tools.default_dtype(calc.default_dtype):
        graphs = [mace_data.AtomicData.from_config(mace_data.config_from_atoms(a, key_specification=keyspec, head_name=calc.head),
                                                   z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)
                  for a in atoms_list]
    batch = torch_geometric.Batch.from_data_list(graphs).to(calc.device)
    out = calc.models[0](batch.to_dict(), compute_stress=False, training=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("xyz_dir"); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--species", default="C10H8_330,C24H12_18,C54H18_37,C96H26_9397")
    a = ap.parse_args()
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import MACECalculator
    print(f"torch {torch.__version__}, GPU {torch.cuda.get_device_name(0)}")
    for dtype in ("float32", "float64"):
        t0 = time.time()
        calc = MACECalculator(model_paths=a.model, device="cuda", default_dtype=dtype, head="omol")
        print(f"\n== {dtype}: model loaded in {time.time() - t0:.1f} s; r_max {calc.r_max}; heads {calc.available_heads}")
        for sp in a.species.split(","):
            for q, m in ((0, 1), (1, 2)):
                f = f"{a.xyz_dir}/{sp}_q{q}.xyz"
                if not os.path.exists(f):
                    print(f"   {sp} q={q}: no geometry"); continue
                at = load_xyz(f); at.info["charge"] = q; at.info["spin"] = m; at.calc = calc
                torch.cuda.reset_peak_memory_stats()
                e = at.get_potential_energy(); keys = sorted(calc.results.keys())
                MaxwellBoltzmannDistribution(at, temperature_K=300, rng=np.random.default_rng(0))
                md = VelocityVerlet(at, 0.5 * units.fs); md.run(5)
                torch.cuda.synchronize(); t = time.time(); md.run(a.steps); torch.cuda.synchronize()
                ms = (time.time() - t) / a.steps * 1e3
                mem = torch.cuda.max_memory_allocated() / 2**30
                extra = ""
                if q == 1:                                       # spin convention check: doublet vs singlet energy
                    b = load_xyz(f); b.info["charge"] = 1; b.info["spin"] = 1; b.calc = calc
                    extra = f"; E(spin=1) - E(spin=2) {b.get_potential_energy() - e:+.3f} eV"
                print(f"   {sp:12s} q={q} {len(at):3d} atoms: {ms:7.2f} ms per step ({ms * 2:.0f} s per ps), "
                      f"peak {mem:.2f} GB; outputs {keys}{extra}")
                # batched throughput, same molecule, displaced copies
                rng = np.random.default_rng(1); line = []
                for nb in (1, 8, 32, 64):
                    lst = []
                    for _ in range(nb):
                        c = at.copy(); c.positions += rng.normal(0, 0.02, c.positions.shape)
                        c.info["charge"] = q; c.info["spin"] = m; lst.append(c)
                    try:
                        batched_eval(calc, lst); torch.cuda.synchronize(); t = time.time()
                        for _ in range(5):
                            out = batched_eval(calc, lst)
                        torch.cuda.synchronize(); dt = (time.time() - t) / 5
                        line.append(f"{nb}: {dt / nb * 1e3:.2f} ms/mol")
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as err:   # TorchScript wraps OOM in RuntimeError
                        if "out of memory" not in str(err):
                            raise
                        line.append(f"{nb}: OOM"); out = {}; torch.cuda.empty_cache(); break
                    finally:
                        torch.cuda.empty_cache()
                print(f"      batched (energy+forces per molecule): " + "; ".join(line) + f"; batch outputs {sorted(k for k, v in out.items() if v is not None)}")
                del out; torch.cuda.empty_cache()
        del calc; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
