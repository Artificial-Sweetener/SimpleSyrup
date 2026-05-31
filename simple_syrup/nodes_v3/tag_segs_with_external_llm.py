# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comfy v3 node for tagging existing SEGS with an external LLM."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.external_llm import (
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
    EXTERNAL_LLM_REASONING_EFFORTS,
)
from ..nodes import tooltips
from ..runtime.external_llm_images import SEG_IMAGE_MODES
from ..services.tag_segs_with_external_llm_service import (
    LLMTagFormattingControls,
    TagSEGSWithExternalLLMService,
)

MAX_EXTERNAL_LLM_MAX_TOKENS = 32768
DEFAULT_SEG_IMAGE_MODE = "transparent mask"

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io
ConditioningBatchIO: Any = (
    None if TYPE_CHECKING else _comfy_io.Custom("CONDITIONING_BATCH")
)


class TagSEGSWithExternalLLMV3(_ComfyNodeBase):
    """Expose external-LLM SEGS tagging through Comfy's v3 API."""

    _service = TagSEGSWithExternalLLMService()

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Tag SEGS w/ External LLM v3 schema."""

        model_choices = cls._service.model_choices()
        return _comfy_io.Schema(
            node_id="SimpleSyrup.TagSEGSWithExternalLLM",
            display_name="Tag SEGS w/ External LLM",
            category="SimpleSyrup/Detailing",
            description=(
                "Tags existing SEGS crops with a configured external vision LLM "
                "and returns aligned conditioning for SEGS detailing."
            ),
            search_aliases=[
                "llm",
                "vision",
                "tag",
                "segs",
                "detail",
                "regional",
                "prompt",
            ],
            inputs=[
                _comfy_io.Image.Input(
                    "image",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_IMAGE,
                ),
                _comfy_io.SEGS.Input(
                    "segs",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_SEGS,
                ),
                _comfy_io.Clip.Input(
                    "clip",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_CLIP,
                ),
                _comfy_io.Combo.Input(
                    "model",
                    options=model_choices,
                    default=model_choices[0],
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_MODEL,
                ),
                _comfy_io.String.Input(
                    "system_prompt",
                    multiline=False,
                    default="",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_SYSTEM_PROMPT,
                ),
                _comfy_io.String.Input(
                    "user_prompt",
                    multiline=False,
                    default="",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_USER_PROMPT,
                ),
                _comfy_io.String.Input(
                    "universal_positive",
                    multiline=False,
                    default="",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_UNIVERSAL_POSITIVE,
                ),
                _comfy_io.Combo.Input(
                    "seg_image_mode",
                    options=list(SEG_IMAGE_MODES),
                    default=DEFAULT_SEG_IMAGE_MODE,
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_IMAGE_MODE,
                ),
                _comfy_io.Boolean.Input(
                    "replace_underscore",
                    default=True,
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_REPLACE_UNDERSCORE,
                ),
                _comfy_io.Boolean.Input(
                    "trailing_comma",
                    default=False,
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_TRAILING_COMMA,
                ),
                _comfy_io.String.Input(
                    "exclude_tags",
                    multiline=False,
                    default="",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_EXCLUDE_TAGS,
                ),
                _comfy_io.Int.Input(
                    "max_tokens",
                    default=DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
                    min=1,
                    max=MAX_EXTERNAL_LLM_MAX_TOKENS,
                    step=1,
                    tooltip=tooltips.EXTERNAL_LLM_MAX_TOKENS_INPUT,
                ),
                _comfy_io.Combo.Input(
                    "reasoning_effort",
                    options=list(EXTERNAL_LLM_REASONING_EFFORTS),
                    default=DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
                    tooltip=tooltips.EXTERNAL_LLM_REASONING_EFFORT_INPUT,
                ),
            ],
            outputs=[
                _comfy_io.SEGS.Output(
                    "segs",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_SEGS_OUTPUT,
                ),
                ConditioningBatchIO.Output(
                    "positive",
                    tooltip=tooltips.EXTERNAL_LLM_TAG_SEGS_POSITIVE_OUTPUT,
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: object,
        segs: object,
        clip: Any,
        model: str,
        system_prompt: str,
        user_prompt: str,
        universal_positive: str,
        seg_image_mode: str,
        replace_underscore: bool,
        trailing_comma: bool,
        exclude_tags: str,
        max_tokens: int = DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
        reasoning_effort: str = DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
    ) -> tuple[object, object]:
        """Tag existing SEGS and return aligned conditioning."""

        result = cls._service.tag(
            image=image,
            segs=segs,
            clip=clip,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            universal_positive=universal_positive,
            seg_image_mode=seg_image_mode,
            formatting=LLMTagFormattingControls(
                replace_underscore=replace_underscore,
                trailing_comma=trailing_comma,
                exclude_tags=exclude_tags,
            ),
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )
        return result.segs, result.positive
