# T0 timing grid: summary

Source: `reports/t0_timing_full.csv`, 93 rows, 31 configurations, 2026-11-09T07:34 to 2026-11-10T11:47. Written by `t0/summarize.py`.

Cost is wall-clock seconds per molecule-picosecond (lower is faster): median of the repeats, with the range (min-max) as spread. Float32, TF32 off in both stacks; GPU 0 and NUMA node 0 of the GH200.

## Decision numbers at batch 100 (plan 4.5)

Fastest PyTorch mode: **TorchSim batched, cueq on**, 0.5132 s/mol-ps (range 0.5052-0.5169).

| JAX configuration | s/mol-ps | speed-up vs fastest PyTorch | band | speed-up vs fastest PyTorch with cueq off |
| --- | --- | --- | --- | --- |
| JAX cueq conv_fusion | 0.1704 | **3.01x** | 2-5x: JAX for the MD loop, PyTorch kept as reference | 6.75x (TorchSim batched, cueq off) |
| JAX cueq backend only | 0.2328 | **2.20x** | 2-5x: JAX for the MD loop, PyTorch kept as reference | 4.94x (TorchSim batched, cueq off) |
| JAX cueq off | 0.2326 | **2.21x** | 2-5x: JAX for the MD loop, PyTorch kept as reference | 4.94x (TorchSim batched, cueq off) |

Worst and best case from the repeat ranges (slowest JAX against fastest PyTorch repeat, and the reverse): JAX cueq conv_fusion 2.96x-3.03x; JAX cueq off 2.17x-2.22x; JAX cueq backend only 2.17x-2.22x. No ratio crosses a band boundary.

Which JAX row the decision uses depends on the committee's definition of JAX "cuEquivariance on" (pending, 2026-11-09; plan 4.4). Caveats required by plan 4.5: the GH200's NVLink-C2C link favours the PyTorch baseline relative to the production cluster; the result depends on which cuEquivariance setting each stack ran (JAX: `cuex` methods traced per row; PyTorch: MACE with `enable_cueq`).

## All configurations

| batch | configuration | n | median s/mol-ps | range | spread | ns/day per traj | GPU util % | GPU mem GB | compile s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | JAX cueq conv_fusion | 3 | 0.7113 | 0.7078-0.7138 | 0.8% | 121 | 21.3 | 0.142 | 3.25 |
| 1 | JAX cueq off | 3 | 0.8283 | 0.8215-0.8420 | 2.5% | 104 | 25.9 | 0.142 | 3.47 |
| 1 | JAX cueq backend only | 3 | 0.8341 | 0.8296-0.8347 | 0.6% | 104 | 27.8 | 0.142 | 3.45 |
| 1 | ASE concatenated, cueq off | 3 | 32.5000 | 32.0911-32.6940 | 1.9% | 2.66 | 14.4 | 0.064 |  |
| 1 | ASE sequential, cueq off | 3 | 33.2859 | 31.8019-33.3161 | 4.5% | 2.6 | 14.4 | 0.064 |  |
| 1 | TorchSim batched, cueq off | 3 | 38.7795 | 37.9099-39.2018 | 3.3% | 2.23 | 13.7 | 0.066 |  |
| 1 | ASE concatenated, cueq on | 3 | 43.0829 | 42.9435-44.3258 | 3.2% | 2.01 | 8.3 | 0.044 |  |
| 1 | ASE sequential, cueq on | 3 | 43.2483 | 42.8927-43.7667 | 2.0% | 2 | 8.3 | 0.044 |  |
| 1 | TorchSim batched, cueq on | 3 | 48.8160 | 48.5933-48.9168 | 0.7% | 1.77 | 8.7 | 0.05 |  |
| 10 | JAX cueq conv_fusion | 3 | 0.2002 | 0.2000-0.2003 | 0.1% | 43.2 | 42.7 | 0.318 | 2.94 |
| 10 | JAX cueq backend only | 3 | 0.2451 | 0.2450-0.2457 | 0.3% | 35.2 | 49.3 | 0.318 | 3.12 |
| 10 | JAX cueq off | 3 | 0.2456 | 0.2455-0.2457 | 0.1% | 35.2 | 46.7 | 0.318 | 3.15 |
| 10 | ASE concatenated, cueq off | 3 | 3.4733 | 3.4044-3.4856 | 2.3% | 2.49 | 28.7 | 0.387 |  |
| 10 | TorchSim batched, cueq off | 3 | 3.8259 | 3.8170-3.8931 | 2.0% | 2.26 | 27.3 | 0.388 |  |
| 10 | ASE concatenated, cueq on | 3 | 4.5530 | 4.5138-4.5816 | 1.5% | 1.9 | 9.8 | 0.138 |  |
| 10 | TorchSim batched, cueq on | 3 | 5.0178 | 4.9625-5.0334 | 1.4% | 1.72 | 10.3 | 0.143 |  |
| 10 | ASE MPS pool P=4, cueq off (includes equilibration) | 3 | 10.5729 | 10.4804-10.8448 | 3.4% | 0.817 | 47.5 | 0 |  |
| 10 | ASE MPS pool P=4, cueq on (includes equilibration) | 3 | 14.3216 | 14.2188-14.3261 | 0.7% | 0.603 | 29 | 0 |  |
| 10 | ASE sequential, cueq off | 3 | 32.2010 | 32.1737-32.3269 | 0.5% | 0.268 | 15.4 | 0.083 |  |
| 10 | ASE sequential, cueq on | 3 | 43.6508 | 43.4333-44.1040 | 1.5% | 0.198 | 8.6 | 0.05 |  |
| 100 | JAX cueq conv_fusion | 3 | 0.1704 | 0.1704-0.1707 | 0.2% | 5.07 | 85.2 | 2.16 | 3.23 |
| 100 | JAX cueq off | 3 | 0.2326 | 0.2323-0.2326 | 0.1% | 3.71 | 89 | 3.87 | 3.42 |
| 100 | JAX cueq backend only | 3 | 0.2328 | 0.2326-0.2328 | 0.1% | 3.71 | 88.9 | 3.87 | 3.42 |
| 100 | TorchSim batched, cueq on | 3 | 0.5132 | 0.5052-0.5169 | 2.3% | 1.68 | 35.8 | 1.88 |  |
| 100 | ASE concatenated, cueq on | 3 | 0.7328 | 0.7322-0.7348 | 0.4% | 1.18 | 24.3 | 1.88 |  |
| 100 | TorchSim batched, cueq off | 3 | 1.1498 | 1.1495-1.1511 | 0.1% | 0.751 | 87.3 | 5.84 |  |
| 100 | ASE concatenated, cueq off | 3 | 1.3721 | 1.3715-1.3798 | 0.6% | 0.63 | 72.9 | 5.84 |  |
| 100 | ASE MPS pool P=32, cueq on (includes equilibration) | 3 | 2.2209 | 2.2104-2.2217 | 0.5% | 0.389 | 83.7 | 0 |  |
| 100 | ASE MPS pool P=32, cueq off (includes equilibration) | 3 | 2.6560 | 2.6483-2.6567 | 0.3% | 0.325 | 94.5 | 0 |  |
| 100 | ASE MPS pool P=16, cueq off (includes equilibration) | 3 | 3.2015 | 3.1860-3.2201 | 1.1% | 0.27 | 87.4 | 0 |  |
| 100 | ASE MPS pool P=16, cueq on (includes equilibration) | 3 | 3.4142 | 3.3970-3.4186 | 0.6% | 0.253 | 79.2 | 0 |  |
| 100 | ASE sequential, cueq off | (extrapolated from batch 10) | 32.2010 | | | | | | |
| 100 | ASE sequential, cueq on | (extrapolated from batch 10) | 43.6508 | | | | | | |

MPS-pool timings include the 2,000 equilibration steps (about 9% of each run; md_torch.py), unlike every other mode; the committee (2026-11-10) asked for them to be labelled, not rerun. gpu_util is averaged over the whole process, model load and equilibration included. GPU memory for the MPS pool is not captured per process (shown as 0). Sequential ASE was not run at batch 100 (plan 4.3); its row is the batch-10 per-molecule cost.
