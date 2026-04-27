"""DEPRECATED: Pipeline execution resources.

GPU lock and CPU executor are now injected via constructor args
to GenerationStrategy (app/strategies/base.py). This module is
kept only for backward compatibility — new code should not use it.
"""

from __future__ import annotations

import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple


_GPU_LOCK: asyncio.Lock | None = None
_CPU_EXECUTOR: ThreadPoolExecutor | None = None


def init_executors(gpu_lock: asyncio.Lock, cpu_executor: ThreadPoolExecutor) -> None:
    """DEPRECATED: Executors are now injected via DI."""
    warnings.warn(
        "init_executors() is deprecated. Executors are injected via strategy constructor.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _GPU_LOCK, _CPU_EXECUTOR
    _GPU_LOCK = gpu_lock
    _CPU_EXECUTOR = cpu_executor


def get_executors() -> Tuple[asyncio.Lock, ThreadPoolExecutor]:
    """DEPRECATED: Use self._gpu_lock / self._cpu_executor on the strategy."""
    warnings.warn(
        "get_executors() is deprecated. Use strategy constructor injection.",
        DeprecationWarning,
        stacklevel=2,
    )
    if _GPU_LOCK is None or _CPU_EXECUTOR is None:
        raise RuntimeError(
            "Executors not initialized. Call init_executors() during startup."
        )
    return _GPU_LOCK, _CPU_EXECUTOR
