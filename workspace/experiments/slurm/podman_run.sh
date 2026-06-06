#!/usr/bin/env bash
# Run a python script inside the AccelForge container via podman.
# Pulls the image if missing (podman storage is node-local on /scratch).
# Usage: podman_run.sh experiments/<script>.py [args...]
set -euo pipefail

IMAGE="${AF_IMAGE:-docker.io/timeloopaccelergy/accelforge:latest-amd64}"
REPO_WORKSPACE="${REPO_WORKSPACE:-/home/mihika/65930-final/workspace}"

# Pick a node-local, writable base for podman storage. $HOME is NFS and must be
# avoided. /scratch/podman is the intended location but doesn't exist on every node
# (/scratch root isn't writable), so fall back to /tmp (big ext4 on these nodes).
# srun also propagates the login node's XDG_RUNTIME_DIR which is unwritable here.
PODMAN_BASE=""
for base in "/scratch/podman/$USER" "/tmp/podman-$USER"; do
    if mkdir -p "$base/storage" "$base/run" "$base/tmp" "$base/home" 2>/dev/null; then
        PODMAN_BASE="$base"
        break
    fi
done
if [ -z "$PODMAN_BASE" ]; then echo "[podman_run] no writable podman base found" >&2; exit 1; fi
echo "[podman_run] node=$(hostname) podman_base=$PODMAN_BASE"
export STORAGE="$PODMAN_BASE/storage"
export RUNROOT="$PODMAN_BASE/run"
export TMPDIR="$PODMAN_BASE/tmp"
export HOME="${PODMAN_HOME:-$PODMAN_BASE/home}"
export XDG_RUNTIME_DIR="$RUNROOT"
mkdir -p "$STORAGE" "$RUNROOT" "$TMPDIR" "$HOME"

PODMAN="podman --root $STORAGE --runroot $RUNROOT"

if ! $PODMAN image exists "$IMAGE"; then
    echo "[podman_run] pulling $IMAGE ..."
    $PODMAN pull "$IMAGE"
fi

# Interpreter in the image is python3 (no `python`); run via a login shell so any
# image env setup is applied, then exec python3 with our args.
exec $PODMAN run --rm \
    -v "$REPO_WORKSPACE":/home/workspace \
    -w /home/workspace \
    -e AF_JOBS="${AF_JOBS:-$(nproc)}" \
    -e SHARD_ID="${SHARD_ID:-0}" \
    -e SHARD_COUNT="${SHARD_COUNT:-1}" \
    "$IMAGE" \
    bash -lc 'exec python3 "$@"' python3 "$@"
