#!/bin/bash
# Opening-shot planner page, computed on this computer (python -m wt_overlay.planner_web).
cd "$(dirname "$0")" || exit 1
[ -f .gitmodules ] && command -v git >/dev/null && git submodule update --init missile_sim
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m wt_overlay.planner_web "$@"
fi
exec python3 -m wt_overlay.planner_web "$@"
