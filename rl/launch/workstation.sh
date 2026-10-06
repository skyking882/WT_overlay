#!/bin/bash
# Start (or continue) a workstation run, detached from the terminal.
#     rl/launch/workstation.sh [run-name] [extra rl.train arguments, e.g. --set ppo.clip=0.1]
# Hard cap: 48 cores in total for learner + workers (config preset "workstation": max_cores=48,
# enforced with CPU affinity at start-up). Never run this on sharedhost.
set -u
case "$(hostname -s)" in
  sharedhost*) echo "refusing to run on $(hostname -s)" >&2; exit 1 ;;
esac
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LB="${LB_BUNDLE:-$HOME/lb_bundle}"
NAME="${1:-workstation}"
RUN_DIR="${RL_RUN_DIR:-$HOME/rl_runs/$NAME}"
mkdir -p "$RUN_DIR"
cd "$REPO"
SETS=(--set "workers.python=['$LB/pypy/bin/pypy3']")
setsid nohup "$LB/mlenv/bin/python" -m rl.train run --config workstation --run-dir "$RUN_DIR" \
    "${SETS[@]}" "${@:2}" > "$RUN_DIR/train.log" 2>&1 < /dev/null &
echo "started pid $! ; run dir $RUN_DIR ; log $RUN_DIR/train.log ; metrics $RUN_DIR/metrics.jsonl"
echo "stop cleanly (checkpoints first):  kill -USR1 $!   (then run this script again to continue)"
