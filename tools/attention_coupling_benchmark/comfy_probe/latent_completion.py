# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Synchronize one latent result without cloning its prepared model."""

from __future__ import annotations

import time
from importlib import import_module
from typing import TYPE_CHECKING, Any

import torch

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for the benchmark-only Comfy v3 node."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


class CompleteLatentV3(_ComfyNodeBase):
    """Expose a synchronized completion timestamp for an unchanged latent."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the benchmark-only latent completion terminal."""

        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.CompleteLatent",
            display_name="Benchmark Complete Latent",
            category="SimpleSyrup/Benchmark",
            inputs=[_comfy_io.Latent.Input("latent")],
            outputs=[_comfy_io.Latent.Output("latent")],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(cls, latent: dict[str, Any]) -> Any:
        """Synchronize queued CUDA work and publish its completion time."""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        completed_at_ns = time.perf_counter_ns()
        return _comfy_io.NodeOutput(
            latent,
            ui={"benchmark_completion": [{"completed_at_ns": completed_at_ns}]},
        )
