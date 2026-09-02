"""Entry point for ``python serve.py ...``, so the service runs on a fresh
clone with no ``pip install -e .``. The implementation is in ``anom_detect.serve``.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).with_name("src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from anom_detect.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
