# Container C on an x86-64 (Intel or AMD) CPU-only machine

Container C (PySR, PAHFIT, AmesPAHdbPythonSuite) needs no GPU. `C_cpu_x86.def` rebuilds it for x86-64 with
the package versions pinned to the GH200 image `C_cpu.sif` (built 2026-11-07), so slice runs on either machine
use the same software.

An Apptainer image runs only on the CPU architecture it was built for. Build this one **on an x86-64 machine**:
the recipe refuses to build anywhere else. Building it on the GH200 would need CPU emulation, under which Julia's
precompilation is impractically slow.

## Files

| File | What it is |
| --- | --- |
| `C_cpu_x86.def` | The x86-64 recipe |
| `C_cpu_lock/requirements-pypi.txt` | The 145 PyPI versions frozen in `C_cpu.sif` (from `requirements-C_cpu-container.txt`). PAHFIT and the PAHdb suite are left out because the recipe installs them from the same git commits as `C_cpu.def`. |
| `C_cpu_lock/Project.toml`, `C_cpu_lock/Manifest.toml` | The Julia environment copied from `C_cpu.sif`: Julia 1.13.3, SymbolicRegression 2.7.x, PythonCall 0.9.36, OpenSSL_jll 3.5.8 |
| `tests/run_c.sh` | The container-C checks on their own, with no GPU and no NUMA pinning |

## Requirements on the x86 machine

- x86-64 Linux with Apptainer (1.3 or later is fine). `--fakeroot` needs either a setuid installation or
  unprivileged user namespaces; otherwise ask an administrator to build it, or build with `sudo`.
- Internet access during the build: Ubuntu packages, PyPI, GitHub, and the Julia package server.
- About 5 GB of free local disk for the build. The image itself is about 1.3 GB.

## Build

Run it from the repository root, because the recipe's `%files` path (`containers/C_cpu_lock`) is relative to it:

```bash
cd pah_check01            # the repository root
export APPTAINER_TMPDIR=/path/to/local/disk/tmp   # optional: avoid a small /tmp or a network file system
apptainer build --fakeroot C_cpu_x86.sif containers/C_cpu_x86.def
```

The build takes roughly 20–40 minutes. Most of it is Julia precompiling SymbolicRegression, done twice:
once when juliapkg sets up the environment, and once after the locked Manifest replaces the one juliapkg
wrote. The build stops with an error if any of these fails:

- `pip versions match the lock: 145 packages`: every PyPI package has the GH200 version.
- `Project.toml matches the lock`.
- `Julia 1.13.3: package versions match the lock`: every Julia package has the GH200 version.
- no `/opt/julia_env/.CondaPkg`: PythonCall uses the venv's Python, not a second conda Python.
- `import pysr` works offline without juliapkg resolving again.

## Test

```bash
containers/tests/run_c.sh C_cpu_x86.sif            # optional second argument: work directory
```

It prints the CPU model, the Python/OpenSSL/Julia versions, and then runs:

1. `c_tools.py`: PySR serial deterministic and multiprocessing fits, the PAHFIT packs (53 classic features),
   and the PAHdb suite import.
2. `c_pahfit.py`: a PAHFIT closure fit with `maxiter=10000`, single-threaded, plus 5 Monte Carlo refits.

Expected results, which match `C_cpu.sif` on the GH200:

- `pysr 2.7.2 import ~4 s`.
- `serial: x0*x0 + cos(x1)*2.5 - 0.5`.
- `multiprocessing:` the same expression, possibly with the last digits of a constant different (this mode is
  not deterministic).
- `classic features: 53`.
- The closure fit recovers 6.2/11.2 ≈ 1.475 against a true value of 1.479.

If the **first** `import pysr` takes minutes rather than seconds, Julia is recompiling, because its precompiled
code did not match this CPU. It does this only once (into `$SCRATCH/julia_depot`, or `$HOME/julia_depot`), so it
is harmless. If it happens on every run, report the CPU model printed by `run_c.sh`.

## Running it

Use the same rules as `C_cpu.sif` (CLAUDE.md, "Non-obvious constraints"):

- **juliapkg takes a lock file** inside `/opt/julia_env`. Bind a writable copy over it, made once per image:
  ```bash
  export SCRATCH=/path/to/local/scratch
  apptainer exec C_cpu_x86.sif cp -r /opt/julia_env "$SCRATCH/julia_env_x86"
  apptainer exec --containall --env SCRATCH="$SCRATCH" --bind "$SCRATCH" \
      --bind "$SCRATCH/julia_env_x86:/opt/julia_env" --bind "$PWD" --pwd "$PWD" C_cpu_x86.sif python my_script.py
  ```
  Don't share that copy with the aarch64 image: the Julia installed inside it is architecture-specific.
- **PAHdb:** pass `--env TMPDIR=<local disk>` when loading it (the suite caches a 267 MB pickle there).
- **PySR:** run it unweighted, and set `procs` together with `parallelism="multiprocessing"`.
- **PAHFIT:** fit with `maxiter=10000` and check `model.fitter.message`.

## Comparing x86 and GH200 results

Expect differences at the rounding level. The two architectures use different BLAS kernels and vector widths,
so PAHFIT fits agree to many digits but not bit-for-bit. PySR searches with `parallelism="multiprocessing"` are
random even with a fixed seed, and the same applies across machines. For a cross-machine check, use
`deterministic=True, parallelism="serial"`, or compare the final Pareto-front equations and their losses, not
the search history.

## Verification so far

On 2026-11-09 the recipe was built on the GH200 with only two changes: the architecture guard removed, and
`JULIA_CPU_TARGET=native`. All the lock checks passed, and `run_c.sh` gave the results above, identical to
`C_cpu.sif`. Not yet tested on real x86 hardware:

- that every pinned version has an x86-64 wheel on PyPI;
- that the multi-target Julia code loads without recompiling.

Record the first x86 build here: date, machine, CPU, and the `run_c.sh` output.
