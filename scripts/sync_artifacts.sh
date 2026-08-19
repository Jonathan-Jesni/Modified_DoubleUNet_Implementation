#!/usr/bin/env bash
# Sync BUSI/CBIS run LOGS to a separate, disposable artifacts repository.
#
#   bash scripts/sync_artifacts.sh once      # single sync, verbose - use this to test
#   bash scripts/sync_artifacts.sh loop &    # unattended, every 15 minutes
#
# Cloud instances for this project are ephemeral and every previous run's artifacts
# were lost to a reset. This copies only the small text artifacts (a few hundred KB)
# that are needed to diagnose a run; checkpoints are far too large for git and are
# pulled manually instead.
#
# IMPORTANT: this writes ONLY to $ARTIFACTS. It never stages, commits, or pushes the
# project repository - that history is maintained by hand.
#
# The artifacts remote is NOT hardcoded here: pushes go to whatever `origin` the
# clone at $ARTIFACTS points to. Verify it before trusting the loop:
#
#   git -C ~/modified-doubleunet-run-artifacts remote -v
#   # expected: origin  https://github.com/Jonathan-Jesni/modified-doubleunet-run-artifacts.git

set -uo pipefail

PROJECT="${PROJECT:-$HOME/workspace/Modified_DoubleUNet_Implementation}"
ARTIFACTS="${ARTIFACTS:-$HOME/modified-doubleunet-run-artifacts}"
INTERVAL="${INTERVAL:-900}"
SESSION="${SESSION:-$(date -u +%Y%m%dT%H%M%SZ)}"

sync_once() {
    local dest="$ARTIFACTS/$SESSION"

    if [ ! -d "$ARTIFACTS/.git" ]; then
        echo "FATAL: $ARTIFACTS is not a git clone; see the header of this script"
        return 1
    fi
    if [ ! -d "$PROJECT/runs" ]; then
        echo "note: $PROJECT/runs does not exist yet - nothing to sync"
        return 0
    fi

    mkdir -p "$dest"
    # Text artifacts only. Never checkpoints (*.pt is ~341 MB per epoch).
    (
        cd "$PROJECT" && find runs -type f \
            \( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \) \
            -exec cp --parents {} "$dest/" \;
    )

    git -C "$ARTIFACTS" add -A
    if git -C "$ARTIFACTS" diff --cached --quiet; then
        echo "no change at $(date -u +%H:%M:%SZ)"
        return 0
    fi
    git -C "$ARTIFACTS" commit -m "sync $SESSION $(date -u +%H:%M:%SZ)" || return 1
    git -C "$ARTIFACTS" push origin HEAD || { echo "PUSH FAILED"; return 1; }
    echo "synced at $(date -u +%H:%M:%SZ)"
}

case "${1:-loop}" in
    once)
        sync_once
        ;;
    loop)
        echo "syncing $PROJECT/runs -> $ARTIFACTS/$SESSION every ${INTERVAL}s"
        while true; do
            sync_once || echo "sync error (continuing)"
            sleep "$INTERVAL"
        done
        ;;
    *)
        echo "usage: $0 [once|loop]"
        exit 2
        ;;
esac
