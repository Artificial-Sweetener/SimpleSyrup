# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Measure benchmark phases around exact delegates with device synchronization."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch

_LOGGER = logging.getLogger(
    "simple_syrup.runtime.regional_lora.standard_unet_cold_path"
)


@contextmanager
def measure_synchronized_phase(
    stage: str,
    *,
    device: torch.device | None,
) -> Iterator[None]:
    """Emit one synchronized phase only while cold capture enables DEBUG."""

    if not _LOGGER.isEnabledFor(logging.DEBUG):
        yield
        return
    synchronize_device(device)
    started_at_ns = time.perf_counter_ns()
    try:
        yield
    finally:
        synchronize_device(device)
        _LOGGER.debug(
            "Measured benchmark Attention Coupling phase",
            extra={
                "cold_path_diagnostics": {
                    "stage": stage,
                    "elapsed_ms": (time.perf_counter_ns() - started_at_ns)
                    / 1_000_000.0,
                }
            },
        )


def model_device(model: object) -> torch.device | None:
    """Return a valid Comfy load device when the object exposes one."""

    device = getattr(model, "load_device", None)
    return device if isinstance(device, torch.device) else None


def synchronize_device(device: torch.device | None) -> None:
    """Synchronize only a live CUDA device explicitly owned by the phase."""

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
