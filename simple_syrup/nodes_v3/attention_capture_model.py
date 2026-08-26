# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Internal Comfy v3 MODEL derivation node injected by prompt provenance."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..services.attention_capture_model_service import ATTENTION_CAPTURE_MODEL_SERVICE

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class AttentionCaptureModelV3(_ComfyNodeBase):
    """Derive an observation-only MODEL from a prompt-injected capture plan."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the dev-only internal capture node contract."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.AttentionCaptureModel",
            display_name="Attention Capture Model (Internal)",
            category="SimpleSyrup/Internal",
            description=(
                "Internal prompt-scoped MODEL observer used by downstream "
                "attention-region nodes."
            ),
            is_dev_only=True,
            inputs=[
                _comfy_io.Model.Input(
                    "model",
                    tooltip="Upstream MODEL cloned for observation-only capture.",
                ),
                _comfy_io.Conditioning.Input(
                    "positive",
                    tooltip="Positive conditioning whose prompt tokens are mapped.",
                ),
                _comfy_io.String.Input(
                    "plan_json",
                    multiline=False,
                    tooltip="Prompt-injected capture plan for the target sampler.",
                ),
                _comfy_io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip="Graph-visible CLIP used for exact prompt token mapping.",
                ),
            ],
            outputs=[
                _comfy_io.Model.Output(
                    "model",
                    tooltip="MODEL carrying one observation-only attention observer.",
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        model: object,
        positive: object,
        plan_json: str,
        clip: object | None = None,
    ) -> tuple[object]:
        """Prepare and publish capture state before the target sampler runs."""

        del positive
        return (
            ATTENTION_CAPTURE_MODEL_SERVICE.prepare(
                model=model,
                plan_json=plan_json,
                clip=clip,
            ),
        )
