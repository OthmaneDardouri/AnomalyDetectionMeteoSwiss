"""Entry point for ``python train_test.py ...``, so the CLI runs on a fresh
clone with no ``pip install -e .``. The implementation is in ``anom_detect.cli``.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).with_name("src")
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from anom_detect.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
