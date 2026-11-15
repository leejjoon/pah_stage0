#!/bin/bash
# Restart a preempted GCP Spot VM until the naphthalene DFT MD is done (docs/plan/cloud_aimd_handoff.md).
# Spot VMs created with --instance-termination-action=STOP keep their disk when preempted but stay stopped until
# someone starts them; the systemd unit `aimd` on the VM then resumes the runs from the checkpoints.
# Run where gcloud is logged in (a laptop, any always-on machine), e.g. under nohup or tmux:
#   bash gcp_keepalive.sh VM ZONE [MINUTES=10]
# Every MINUTES: VM stopped -> start it (a start can fail for lack of Spot capacity; it retries next round).
# VM running -> if ~/aimd/run/ALL_DONE exists on it, stop the VM and exit.
set -uo pipefail
VM=${1:?usage: gcp_keepalive.sh VM ZONE [MINUTES]}; ZONE=${2:?ZONE}; every=$(( ${3:-10} * 60 ))
while true; do
  st=$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)' 2>&1)
  case $st in
    TERMINATED|STOPPED)
      echo "$(date '+%F %T') $VM $st: starting"
      gcloud compute instances start "$VM" --zone "$ZONE" --quiet 2>&1 | tail -1 ;;
    RUNNING)
      if gcloud compute ssh "$VM" --zone "$ZONE" --quiet --command 'test -f ~/aimd/run/ALL_DONE' 2>/dev/null; then
        echo "$(date '+%F %T') $VM: all runs done; stopping the VM"
        gcloud compute instances stop "$VM" --zone "$ZONE" --quiet; exit 0
      fi ;;
    *) echo "$(date '+%F %T') $VM: $st" ;;
  esac
  sleep "$every"
done
