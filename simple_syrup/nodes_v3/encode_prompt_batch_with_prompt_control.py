# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for Prompt Control prompt batch encoding."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..runtime.prompt_control_batch_graph import PromptControlBatchGraphBuilder

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

        @classmethod
        def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
            """Return v1-compatible input metadata."""

            raise NotImplementedError

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io
ConditioningBatchIO: Any = (
    None if TYPE_CHECKING else _comfy_io.Custom("CONDITIONING_BATCH")
)


class EncodePromptBatchWithPromptControl(_ComfyNodeBase):
    """Encode separator-delimited prompts with Prompt Control scheduling."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Prompt Control batch encoder schema."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.EncodePromptBatchWithPromptControl",
            display_name="Encode Prompt Batch w/ Prompt Control",
            enable_expand=True,
            category="SimpleSyrup/Conditioning",
            description=(
                "Encodes [SEP]-separated prompts into per-segment Prompt Control "
                "conditioning batches."
            ),
            inputs=[
                _comfy_io.Clip.Input(
                    "clip",
                    raw_link=True,
                    tooltip=(
                        "Prompt Control-ready CLIP connection used to encode each "
                        "prompt entry."
                    ),
                ),
                _comfy_io.String.Input(
                    "positive_prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Positive Prompt Control prompts in SEGS order, separated "
                        "by the separator text."
                    ),
                ),
                _comfy_io.String.Input(
                    "negative_prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Negative Prompt Control prompts in SEGS order, separated "
                        "by the separator text."
                    ),
                ),
                _comfy_io.String.Input(
                    "separator",
                    default="[SEP]",
                    tooltip="Text marker that splits prompts into per-SEGS entries.",
                ),
            ],
            outputs=[
                ConditioningBatchIO.Output(
                    "positive",
                    tooltip="Positive conditioning entries selected by SEGS order.",
                ),
                ConditioningBatchIO.Output(
                    "negative",
                    tooltip="Negative conditioning entries selected by SEGS order.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip: Any,
        positive_prompt: str,
        negative_prompt: str,
        separator: str,
    ) -> Any:
        """Build lazy Prompt Control graph expansion for prompt batches."""

        return PromptControlBatchGraphBuilder().build(
            clip=clip,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            separator=separator,
        )
