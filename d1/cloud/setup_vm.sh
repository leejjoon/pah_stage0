#!/bin/bash
# Set up a cloud VM for the naphthalene DFT MD (docs/plan/cloud_aimd_handoff.md). Run on the VM, as the normal user,
# from the unpacked bundle (aimd_cloud/code/cloud/setup_vm.sh):
#   bash setup_vm.sh install                   copy the bundle to ~/aimd, install micromamba and the pinned env, check it
#   bash setup_vm.sh service [--poweroff-when-done] [THREADS_Q0 THREADS_Q1]
#                                              install + start the systemd unit `aimd` (sudo): resumes the runs on
#                                              every boot; with --poweroff-when-done the VM powers off once all are done
# Layout: ~/aimd/{bin/micromamba, env/, code/{aimd_pyscf.py, cloud/}, run/{naph_*.npz, C10H8_330_q*.xyz, logs/}}
set -euo pipefail
A=$HOME/aimd; SRC=$(cd "$(dirname "$0")/../.." && pwd)    # bundle root: code/ and run/
cmd=${1:?usage: setup_vm.sh install | service [--poweroff-when-done] [THREADS_Q0 THREADS_Q1]}; shift

if [ "$cmd" = install ]; then
  arch=$(uname -m); echo "$(hostname): $arch, $(nproc) CPUs, $(lscpu | sed -n 's/^Model name: *//p' | head -1)"
  mkdir -p "$A/bin"
  if [ "$SRC" != "$A" ]; then   # rerunnable: code is refreshed, checkpoints already in ~/aimd/run are never touched
    rm -rf "$A/code"; cp -a "$SRC/code" "$A/"; echo "code: copied to $A/code"
    if [ -e "$A/run" ]; then echo "run: $A/run exists, kept as it is"; else cp -a "$SRC/run" "$A/"; echo "run: copied"; fi
  fi
  if [ ! -x "$A/bin/micromamba" ]; then
    case $arch in aarch64) mm=linux-aarch64 ;; x86_64) mm=linux-64 ;; *) echo "unsupported $arch"; exit 1 ;; esac
    curl -fsSL -o "$A/bin/micromamba" \
      "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-$mm"
    chmod +x "$A/bin/micromamba"
  fi
  export MAMBA_ROOT_PREFIX=$A/mamba
  if [ ! -x "$A/env/bin/python" ]; then
    if [ "$arch" = aarch64 ]; then   # the project node's env, package for package
      "$A/bin/micromamba" create -y -p "$A/env" --file "$A/code/cloud/dft_env_linux-aarch64.lock"
    else                             # same versions, x86 builds
      "$A/bin/micromamba" create -y -p "$A/env" -c conda-forge --override-channels python=3.12.16 pyscf=2.15.0 \
        ase=3.29.0 numpy=2.5.4 scipy=1.18.1 "libopenblas=0.3.34=*openmp*" "libblas=*=*openblas"
    fi
    "$A/bin/micromamba" clean -y --all >/dev/null
  fi
  "$A/env/bin/python" -I -c "
import pyscf, ase, numpy, scipy; from pyscf import dft, lib
print('pyscf', pyscf.__version__, 'ase', ase.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)
assert (pyscf.__version__, ase.__version__, numpy.__version__) == ('2.15.0', '3.29.0', '2.5.4'), 'version mismatch'
m = pyscf.M(atom='O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59', basis='6-31g*', cart=True, verbose=0)
mf = dft.RKS(m); mf.xc = 'b3lypg'; e = mf.kernel()
print('water B3LYP(VWN3)/6-31G* 6D: %.8f Ha (project node: -76.4088698182)' % e)
assert abs(e + 76.4088698182) < 1e-6, 'energy differs from the project node'"
  ls "$A/run"/naph_*.npz | wc -l | xargs echo "checkpoints:"
  echo "next: the timing check (handoff step 4): $A/code/cloud/launch_aimd.sh timing $A/run"
  exit 0
fi

if [ "$cmd" = service ]; then
  poweroff=""; [ "${1:-}" = --poweroff-when-done ] && { poweroff="SuccessAction=poweroff"; shift; }
  t0=${1:-8}; t1=${2:-8}
  mkdir -p "$A/run/logs"
  sudo tee /etc/systemd/system/aimd.service >/dev/null <<EOF
[Unit]
Description=Naphthalene DFT MD (resumes from the checkpoints on every boot)
After=local-fs.target
$poweroff

[Service]
Type=simple
User=$USER
Environment=THREADS_Q0=$t0 THREADS_Q1=$t1 CKPT_FS=20 AIMD_PY=$A/env/bin/python
ExecStart=$A/code/cloud/launch_aimd.sh run $A/run
# stop (shutdown, Spot preemption): SIGTERM to the launcher only, which passes it to the runs; each saves after its
# current step (at most ~15 s). SIGKILL to whatever is left after 60 s.
KillMode=mixed
TimeoutStopSec=60
StandardOutput=append:$A/run/logs/launcher.log
StandardError=append:$A/run/logs/launcher.log

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload && sudo systemctl enable aimd && sudo systemctl restart aimd   # a running unit saves first
  sleep 5; { systemctl --no-pager status aimd | head -5; } || true; tail -n 12 "$A/run/logs/launcher.log"
  exit 0
fi
echo "unknown command $cmd"; exit 1
