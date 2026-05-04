#!/usr/bin/env bash
# launch_train.sh — start train.py such that it survives:
#   - SIGHUP (terminal close, SSH disconnect)
#   - tmux session crash
#   - logout (when systemd Linger is enabled — see notes below)
#
# Usage:
#   ./scripts/launch_train.sh -- python train.py --algorithm ppo --api ...
#   ./scripts/launch_train.sh --tag medqa-resume -- python train.py ...
#
# Output:
#   PID and log path are printed.
#   Log file: logs/run_<tag>_<timestamp>.out (stdout+stderr, line-buffered)
#   PID file: logs/run_<tag>_<timestamp>.pid
#
# To stop:  kill $(cat <pid_file>)
# To watch: tail -f <log_file>
#
# One-time setup (so tmux/your shell can die without killing training):
#   sudo loginctl enable-linger $USER
#   verify with: loginctl show-user $USER | grep Linger    (should say Linger=yes)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

TAG="run"
# Parse our own flags (everything before --) and pass the rest to nohup.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            # First positional with no -- separator: assume this is the start of the command.
            break
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: no command provided." >&2
    echo "Usage: $0 [--tag <name>] -- <command> [args...]" >&2
    exit 2
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TAG}_${TIMESTAMP}.out"
PID_FILE="$LOG_DIR/run_${TAG}_${TIMESTAMP}.pid"

# Linger advisory — warn but do not block. User can opt out by setting LINGER_OK=1.
if [[ "${LINGER_OK:-0}" != "1" ]]; then
    LINGER_STATUS="$(loginctl show-user "$USER" 2>/dev/null | grep -E '^Linger=' | cut -d= -f2 || echo unknown)"
    if [[ "$LINGER_STATUS" != "yes" ]]; then
        echo "WARNING: systemd Linger is '$LINGER_STATUS' for user $USER."
        echo "  Without Linger, your training process may be killed when you log out of"
        echo "  this machine (e.g., closing the SSH session for the last time)."
        echo "  To enable persistence across logout:  sudo loginctl enable-linger $USER"
        echo "  To suppress this warning:             LINGER_OK=1 $0 ..."
        echo ""
    fi
fi

cd "$REPO_ROOT"

# stdbuf ensures line-buffered stdout so 'tail -f' shows progress immediately.
# nohup detaches from controlling terminal; SIGHUP is ignored.
# `setsid` puts the process in its own session so even SIGHUP-on-process-group is dodged.
nohup setsid stdbuf -oL -eL "$@" >"$LOG_FILE" 2>&1 < /dev/null &
TRAIN_PID=$!

echo "$TRAIN_PID" > "$PID_FILE"

cat <<EOF
========================================================================
Training launched.
  PID:      $TRAIN_PID
  Log:      $LOG_FILE
  PID file: $PID_FILE
  Tag:      $TAG

Useful commands:
  tail -f $LOG_FILE                   # watch progress
  kill \$(cat $PID_FILE)               # stop training
  ps -p $TRAIN_PID -o pid,etime,cmd   # is it still running?
========================================================================
EOF
