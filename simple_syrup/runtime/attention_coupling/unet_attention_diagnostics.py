# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Emit structured standard-UNet regional resolution diagnostics."""

from __future__ import annotations

import logging

from ...shared.logging import get_logger
from ..regional_attention_diagnostic_values import (
    RegionalAttentionExecutionDiagnostics,
)
from .unet_attn2_geometry import StandardUnetAttn2Geometry

LOGGER = get_logger("runtime.attention_coupling.unet_diagnostics")


class StandardUnetAttentionDiagnosticsEmitter:
    """Publish one JSON-safe record for each unique call-local resolution."""

    def __init__(self, logger: logging.Logger = LOGGER) -> None:
        """Retain the sole standard-UNet diagnostics logging boundary."""

        if not isinstance(logger, logging.Logger):
            raise TypeError("Standard UNet diagnostics require a logger.")
        self._logger = logger

    def emit(
        self,
        snapshot: RegionalAttentionExecutionDiagnostics,
        geometry: StandardUnetAttn2Geometry,
    ) -> None:
        """Emit one resolution snapshot without model inputs or tensor values."""

        if not isinstance(snapshot, RegionalAttentionExecutionDiagnostics):
            raise TypeError("Standard UNet diagnostics require a shared snapshot.")
        if not isinstance(geometry, StandardUnetAttn2Geometry):
            raise TypeError("Standard UNet diagnostics require validated geometry.")
        if not self._logger.isEnabledFor(logging.INFO):
            return
        self._logger.info(
            "Standard UNet regional Attention Coupling resolution",
            extra={
                "operation": "unet_attention_coupling.resolve",
                "unet_layer": {
                    "block_kind": geometry.block[0],
                    "block_number": geometry.block[1],
                    "block_index": geometry.block_index,
                    "transformer_index": geometry.transformer_index,
                    "query_height": geometry.query.query_height,
                    "query_width": geometry.query.query_width,
                },
                "regional_diagnostics": snapshot.to_log_fields(),
            },
        )


STANDARD_UNET_ATTENTION_DIAGNOSTICS_EMITTER = StandardUnetAttentionDiagnosticsEmitter()
