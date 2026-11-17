# Parity of a mace-torch model and its mace-jax conversion (mace-jax-from-torch under JAX_ENABLE_X64=1), per head,
# in float64, on the first N frames of an extxyz per head:
#   container B:  python mace_parity.py torch <cpu .model> <out.npz> HEAD=<extxyz> [HEAD=<extxyz> ...] [--n 5]
#   container A:  python mace_parity.py jax <bundle .npz> <torch out.npz> HEAD=<extxyz> ... [--n 5]
# The jax step prints max |dE| (eV) and max |dF| (eV/A) per head against the torch output and the DFT labels, and
# applies the T0 gates (plan §4.2): float64 total energy within 1e-6 eV/atom, forces within 1e-3 eV/A.
import argparse, sys
import numpy as np
from ase.io import read


def frames(spec, n):
    head, path = spec.split("=", 1)
    return head, read(path, f":{n}")


def run_torch(a):
    import torch
    from mace.calculators import MACECalculator
    torch.set_default_dtype(torch.float64)
    out = {}
    for spec in a.heads:
        head, ats = frames(spec, a.n)
        calc = MACECalculator(model_paths=a.model, device="cpu", default_dtype="float64", head=head)
        for k, at in enumerate(ats):
            at.calc = calc
            out[f"{head}/{k}/E"] = at.get_potential_energy(); out[f"{head}/{k}/F"] = at.get_forces()
    np.savez(a.out, **out)
    print(f"torch: {len(out) // 2} frames -> {a.out}")


def run_jax(a):
    import jax, jraph
    from mace_jax import data
    from mace_jax.cli import mace_jax_predict as P
    from mace_jax.tools.bundle import load_model_bundle
    assert jax.config.jax_enable_x64, "run with JAX_ENABLE_X64=1"
    b = load_model_bundle(a.model, "float64"); cfg = b.config
    ref = np.load(a.out); z = data.AtomicNumberTable(cfg["atomic_numbers"])
    pred = jax.jit(P._build_predictor(b.graphdef, model_config=cfg, compute_forces=True, compute_stress=False,
                                      full_outputs=False))
    bad = []
    for spec in a.heads:
        head, ats = frames(spec, a.n)
        head_name, h2i = P._resolve_head_mapping(cfg, head)
        _, confs = data.load_from_xyz(spec.split("=", 1)[1], head_name=head_name, no_data_ok=True)
        dE, dF, eD, fD, dEa = [], [], [], [], []
        for k, at in enumerate(ats):
            g = data.graph_from_configuration(confs[k], cutoff=float(cfg["r_max"]), z_table=z, head_to_index=h2i)
            n = int(g.n_node.sum())
            g = jraph.pad_with_graphs(g, n_node=n + 1, n_edge=int(g.n_edge.sum()), n_graph=2)
            o = pred(b.params, g)
            e = float(np.asarray(o["energy"])[0]); f = np.asarray(o["forces"])[:n]
            dE.append(abs(e - float(ref[f"{head}/{k}/E"]))); dEa.append(dE[-1] / n); dF.append(np.abs(f - ref[f"{head}/{k}/F"]).max())
            eD.append(abs(e - at.info["dft_energy"]) / n); fD.append(np.sqrt(((f - at.arrays["dft_forces"]) ** 2).mean()))
        if max(dEa) > 1e-6 or max(dF) > 1e-3:
            bad.append(head)
        print(f"{head}: {len(ats)} frames; jax vs torch max |dE| {max(dE):.2e} eV ({max(dEa):.1e} eV/atom), max |dF| {max(dF):.2e} eV/A; "
              f"vs DFT mean |dE|/atom {1e3 * np.mean(eD):.1f} meV, mean F RMSE {1e3 * np.mean(fD):.1f} meV/A")
    print("PASS (T0 gates: 1e-6 eV/atom, 1e-3 eV/A)" if not bad else f"FAIL: {bad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("side", choices=["torch", "jax"]); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("heads", nargs="+"); ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    (run_torch if a.side == "torch" else run_jax)(a)


if __name__ == "__main__":
    main()
