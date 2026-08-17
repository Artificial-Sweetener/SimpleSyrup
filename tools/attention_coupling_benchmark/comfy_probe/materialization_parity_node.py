# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose image-free materialization parity as one benchmark-only node."""

from __future__ import annotations

import json
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from .materialization_parity_probe import (
    MATERIALIZATION_PARITY_PROBE,
    MaterializationParityProbe,
)

_comfy_api: Any = None
if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for the benchmark-only node."""

        pass

else:
    _comfy_api = import_module("comfy_api.latest")
    _ComfyNodeBase = _comfy_api.io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else _comfy_api.io


class CompareMaterializationParityV3(_ComfyNodeBase):
    """Publish CPU-versus-selected-device regional parameter-bank evidence."""

    probe: ClassVar[MaterializationParityProbe] = MATERIALIZATION_PARITY_PROBE

    @classmethod
    def define_schema(cls) -> Any:
        """Declare one terminal benchmark comparison without sampler outputs."""

        conditioning_batch = _comfy_io.Custom("CONDITIONING_BATCH")
        return _comfy_io.Schema(
            node_id="SimpleSyrupBenchmark.CompareMaterializationParity",
            display_name="Benchmark Compare Materialization Parity",
            category="SimpleSyrup/Benchmark",
            inputs=[
                _comfy_io.Model.Input("model"),
                _comfy_io.MultiType.Input(
                    "positive",
                    [_comfy_io.Conditioning, conditioning_batch],
                ),
                _comfy_io.MultiType.Input(
                    "negative",
                    [_comfy_io.Conditioning, conditioning_batch],
                ),
                _comfy_io.Mask.Input("region_masks"),
                _comfy_io.Latent.Input("latent_image"),
                _comfy_io.Int.Input(
                    "region_mask_feather",
                    default=0,
                    min=0,
                    max=4096,
                ),
                _comfy_io.String.Input("run_id"),
            ],
            outputs=[_comfy_io.String.Output("comparison_json")],
            is_output_node=True,
            is_dev_only=True,
        )

    @classmethod
    def execute(
        cls,
        model: object,
        positive: object,
        negative: object,
        region_masks: object,
        latent_image: object,
        region_mask_feather: int,
        run_id: str,
    ) -> Any:
        """Run one synchronized comparison and publish JSON-safe evidence."""

        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Materialization parity run id must be non-empty.")
        payload = cls.probe.compare(
            model=model,
            positive=positive,
            negative=negative,
            region_masks=region_masks,
            latent_image=latent_image,
            region_mask_feather=region_mask_feather,
        ).as_json_object()
        payload["run_id"] = run_id
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return _comfy_io.NodeOutput(
            encoded,
            ui={"materialization_parity": [payload]},
        )
