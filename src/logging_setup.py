"""Единая настройка логирования.

В проде логи читает не человек, а сборщик логов, поэтому формат один
на все точки входа и включает время, уровень и модуль.
"""
from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str = None) -> logging.Logger:
    level = level or os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("mlops")
