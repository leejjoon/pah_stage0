#!/bin/bash
# Run md_torch.py --mode mps under a per-user CUDA MPS daemon on GPU 0 (plan 4.3).
#
# The daemon must run INSIDE the same image as the workers. Container B (NGC 26.10) uses the CUDA forward-
# compatibility libcuda (615.71, CUDA 13.4) while the host driver is 595.45 (CUDA 13.2); a host daemon accepts the
# container clients and then their cuInit blocks forever (checked 2026-11-09: host clients on the same daemon work).
# nvidia-cuda-mps-control/-server are host-only binaries, so they are bound into the image; run there, the server
# loads the image's compat libcuda, the same one the clients use.
#
# The daemon runs in the foreground (-f) inside the same CUDA_VISIBLE_DEVICES=0 / numactl pinning as the workers,
# logging to $LOG; the pipe directory is short (Unix socket paths are limited to 108 bytes). After the run the script
# counts the client connections in that log, because a client that cannot reach the daemon silently runs without MPS.
# Usage: t0/run_mps.sh <sif> <md_torch.py args...>   (repo bound at /r, output dir at /md)
set -euo pipefail
SIF=$1; shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
MD=${MD_DIR:-/var/tmp/$USER-pah/md}
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/dev/shm/mps-$USER/pipe CUDA_MPS_LOG_DIRECTORY=/dev/shm/mps-$USER/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$MD"
LOG=$CUDA_MPS_LOG_DIRECTORY/daemon-$(date +%Y%m%d-%H%M%S).log
PIN=(numactl --cpunodebind=0 --membind=0)
BIND=(--bind /usr/bin/nvidia-cuda-mps-control:/usr/bin/nvidia-cuda-mps-control
      --bind /usr/bin/nvidia-cuda-mps-server:/usr/bin/nvidia-cuda-mps-server
      --bind /dev/shm/mps-$USER)

if pgrep -u "$USER" -f "nvidia-cuda-mps-contro[l]" >/dev/null; then echo "an MPS control daemon is already running for $USER"; exit 1; fi
"${PIN[@]}" apptainer exec --nv "${BIND[@]}" "$SIF" nvidia-cuda-mps-control -f > "$LOG" 2>&1 &
DPID=$!
stop_daemon() { apptainer exec --nv "${BIND[@]}" "$SIF" bash -c "echo quit | nvidia-cuda-mps-control" || true; wait "$DPID" 2>/dev/null || true; }
trap stop_daemon EXIT
# the socket file outlives a previous daemon, so wait for this daemon's own start-up line instead
for _ in $(seq 60); do grep -q "Starting control daemon" "$LOG" && break; sleep 0.5; done
grep -q "Starting control daemon" "$LOG" || { echo "MPS control daemon did not start:"; cat "$LOG"; exit 1; }

"${PIN[@]}" apptainer exec --nv "${BIND[@]}" --bind "$REPO":/r --bind "$MD":/md "$SIF" \
  python /r/t0/md_torch.py --mode mps "$@"
echo "MPS daemon log $LOG: $(grep -o "NEW CLIENT [0-9]*" "$LOG" | sort -u | wc -l) distinct client processes during this run"
