"""python -m wt_overlay.planner_web (the guard matters: worker processes import this module again)."""
from .server import main

if __name__ == "__main__":
    raise SystemExit(main())
