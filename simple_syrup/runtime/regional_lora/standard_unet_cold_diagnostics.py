# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Emit opt-in timing evidence for standard-UNet cold-path stages."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum

import torch

ColdMetadataValue = bool | float | int | str
ColdMetadata = dict[str, ColdMetadataValue]
Clock = Callable[[], int]
Synchronizer = Callable[[torch.device | None], None]

_LOGGER = logging.getLogger(
    "simple_syrup.runtime.regional_lora.standard_unet_cold_path"
)


class StandardUnetColdStage(StrEnum):
    """Identify non-overlapping or explicitly aggregate first-use stages."""

    ADMISSION_RESOLUTION = "admission_resolution"
    VARIANT_MATERIALIZATION = "variant_materialization"
    VARIANT_SHELL = "variant_shell"
    TEMPLATE_PREPARATION = "template_preparation"
    MODEL_RESIDENCY = "model_residency"
    SAMPLING = "sampling"


class StandardUnetColdPathDiagnosticsEmitter:
    """Measure exact stage boundaries only while a DEBUG capture is active."""

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        clock_ns: Clock = time.perf_counter_ns,
        synchronize: Synchronizer | None = None,
    ) -> None:
        """Retain injected timing boundaries for deterministic characterization."""

        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("Standard UNet cold diagnostics logger is invalid.")
        if not callable(clock_ns):
            raise TypeError("Standard UNet cold diagnostics clock is invalid.")
        if synchronize is not None and not callable(synchronize):
            raise TypeError("Standard UNet cold diagnostics synchronizer is invalid.")
        self._logger = logger or _LOGGER
        self._clock_ns = clock_ns
        self._synchronize = synchronize or _synchronize_device

    @contextmanager
    def measure(
        self,
        stage: StandardUnetColdStage,
        *,
        device: torch.device | None = None,
    ) -> Iterator[ColdMetadata]:
        """Yield mutable bounded metadata and emit one synchronized observation."""

        if not isinstance(stage, StandardUnetColdStage):
            raise TypeError("Standard UNet cold diagnostic stage is invalid.")
        if device is not None and not isinstance(device, torch.device):
            raise TypeError("Standard UNet cold diagnostic device is invalid.")
        metadata: ColdMetadata = {}
        if not self._logger.isEnabledFor(logging.DEBUG):
            yield metadata
            return
        self._synchronize(device)
        started_at_ns = self._clock_ns()
        try:
            yield metadata
        finally:
            self._synchronize(device)
            elapsed_ms = (self._clock_ns() - started_at_ns) / 1_000_000.0
            self._logger.debug(
                "Measured standard UNet cold-path stage",
                extra={
                    "operation": "standard_unet_cold_path.measure",
                    "cold_path_diagnostics": {
                        "stage": stage.value,
                        "elapsed_ms": elapsed_ms,
                        **metadata,
                    },
                },
            )


def _synchronize_device(device: torch.device | None) -> None:
    """Synchronize only an available CUDA boundary selected by the stage owner."""

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


STANDARD_UNET_COLD_PATH_DIAGNOSTICS = StandardUnetColdPathDiagnosticsEmitter()
