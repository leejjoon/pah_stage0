# Vectorised all-pairs graph for PolarMACE's real-space electrostatics (graph_longrange 0.4.4).
# graph_longrange.realspace_electrostatics.batch_complete_graph_excluding_self_duplicates_vector builds the
# per-molecule complete graph with a Python loop over molecules (a GPU sync per molecule) and is called 5x per forward:
# ~1.3 ms per molecule per call, so batched dipole evaluation does not get cheaper per frame with batch size and
# training is CPU-bound. This replacement builds the same edge list (same order, for PyG batches whose `batch` is
# sorted, as mace's are) with tensor ops only. install() patches the module attribute, which all callers look up.
#   python d1/polar_fastgraph.py check                       # equality test against the original
#   python d1/polar_fastgraph.py train <run_train args...>   # mace.cli.run_train with the patch (torchrun-able)
import sys
import torch


@torch.no_grad()
def complete_graph_excluding_self_duplicates(batch: torch.Tensor, N: int) -> torch.Tensor:
    batch = batch.long()
    orig2 = torch.arange(batch.size(0), device=batch.device).repeat_interleave(N)
    batch2 = batch.repeat_interleave(N)
    if batch2.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=batch.device)
    D = torch.bincount(batch2)                        # duplicated nodes per graph
    start = torch.cumsum(D, 0) - D
    per_row = D[batch2]                               # each node pairs with every node of its graph
    rows = torch.repeat_interleave(torch.arange(batch2.numel(), device=batch.device), per_row)
    row_start = torch.cumsum(per_row, 0) - per_row
    within = torch.arange(rows.numel(), device=batch.device) - torch.repeat_interleave(row_start, per_row)
    cols = start[batch2[rows]] + within
    keep = orig2[rows] != orig2[cols]
    return torch.stack([rows[keep], cols[keep]], dim=0)


def install():
    from graph_longrange import realspace_electrostatics as rs
    rs._orig_complete_graph = getattr(rs, "_orig_complete_graph", rs.batch_complete_graph_excluding_self_duplicates_vector)
    rs.batch_complete_graph_excluding_self_duplicates_vector = complete_graph_excluding_self_duplicates


def check():
    from graph_longrange import realspace_electrostatics as rs
    for sizes in ([24, 24, 3], [1, 5, 126], [126] * 7, [2]):
        b = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
        for N in (1, 4):
            ref = rs.batch_complete_graph_excluding_self_duplicates_vector(b, N)
            new = complete_graph_excluding_self_duplicates(b, N)
            assert torch.equal(ref, new), (sizes, N)
    print("identical edge lists")


if __name__ == "__main__":
    cmd = sys.argv.pop(1)
    if cmd == "check":
        check()
    elif cmd == "train":
        install()
        from mace.cli.run_train import main
        main()
