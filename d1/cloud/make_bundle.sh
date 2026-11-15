#!/bin/bash
# Build the cloud bundle for the naphthalene DFT MD (docs/plan/cloud_aimd_handoff.md), on the project node:
#   bash d1/cloud/make_bundle.sh [CHECKPOINT_TGZ] [OUT_TGZ]
# CHECKPOINT_TGZ: the Hangang inputs tarball (aimd_naph/: 8 npz + 2 xyz; default /var/tmp/jjlee-pah/d1/
# aimd_naph_hangang.tgz) or a later tarball of the same layout, e.g. results handed back from a partial run.
# OUT_TGZ (default /var/tmp/jjlee-pah/d1/aimd_naph_cloud.tgz) holds aimd_cloud/{code/{aimd_pyscf.py, cloud/}, run/}.
# Only the checkpoints, the two PAHdb naphthalene geometries and this code go in: no logs (local paths), no .env,
# no Mai et al. data.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
IN=${1:-/var/tmp/jjlee-pah/d1/aimd_naph_hangang.tgz}; OUT=${2:-/var/tmp/jjlee-pah/d1/aimd_naph_cloud.tgz}
W=$(mktemp -d /var/tmp/jjlee-pah/bundle.XXXXXX); trap 'rm -r -- "$W"' EXIT
mkdir -p "$W/aimd_cloud/code/cloud" "$W/aimd_cloud/run" "$W/in"
tar xzf "$IN" -C "$W/in"
cp "$W"/in/aimd_naph/naph_q[01]_*K_r[0-9].npz "$W"/in/aimd_naph/C10H8_330_q[01].xyz "$W/aimd_cloud/run/"
cp "$REPO/d1/aimd_pyscf.py" "$W/aimd_cloud/code/"
cp "$REPO"/d1/cloud/{launch_aimd.sh,setup_vm.sh,gcp_keepalive.sh,dft_env_linux-aarch64.lock} "$W/aimd_cloud/code/cloud/"
echo "$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet -- d1 || echo '+dirty')" > "$W/aimd_cloud/code/VERSION"
tar czf "$OUT" -C "$W" --owner=0 --group=0 aimd_cloud
tar tzf "$OUT" | sed 's|^|  |'
sha256sum "$OUT"; du -h "$OUT" | cut -f1
