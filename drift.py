"""Entry point for ``python drift.py ...``.

The implementation lives in ``anom_detect.drift``. Prefer ``pip install -e .``
followed by the ``anom-detect-drift`` console script; this file exists so the
drift check runs on a fresh clone without an install, and without depending on
the console script landing somewhere on PATH.

Exits 1 on detected drift when ``--fail-on-drift`` is passed, so it works as a
CI or cron gate either way it is invoked.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).with_name("src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from anom_detect.drift import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
