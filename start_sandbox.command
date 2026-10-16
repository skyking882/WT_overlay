#!/bin/bash
cd "$(dirname "$0")" || exit 1
[ -f .gitmodules ] && command -v git >/dev/null && git submodule update --init
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m wt_overlay.sandbox "$@"
fi
exec python3 -m wt_overlay.sandbox "$@"
