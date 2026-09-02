"""Shared logging setup: every module gets its logger via :func:`get_logger`."""

import logging
from collections.abc import Sequence
from typing import Any

_CONFIGURED = False

_BLOCK_WIDTH = 64


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name``, configuring the root handler on first use."""
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def _format_value(value: Any) -> str:
    """Right-hand column text: floats to 4 decimals, everything else as-is."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def log_results_block(
    logger: logging.Logger, title: str, rows: Sequence[tuple[str, Any]]
) -> None:
    """Log ``rows`` as one bordered label/value block.

    A single record, not one per row, so the timestamp prefix isn't repeated.
    """
    label_width = max((len(label) for label, _ in rows), default=0)
    border = "=" * _BLOCK_WIDTH
    lines = [
        "",
        border,
        f"  {title}",
        "-" * _BLOCK_WIDTH,
        *(f"  {label:<{label_width}}   {_format_value(value)}" for label, value in rows),
        border,
    ]
    logger.info("\n".join(lines))
