# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Application service for external-LLM tagging of existing SEGS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..domain.conditioning_batch import ConditioningBatch
from ..domain.external_llm import (
    DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
    DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
)
from ..domain.prompt_composition import prefix_prompt
from ..domain.segs import (
    ImpactSegs,
    NativeSegs,
    Segment,
    coerce_segs,
    to_impact_compatible_segs,
)
from ..masking.segs_mask_ops import validate_single_image
from ..runtime.conditioning_encoding import ComfyConditioningEncoder
from ..runtime.external_llm_images import ExternalLLMSegsImageEncoder
from ..runtime.progress import ProgressReporter, create_comfy_progress
from ..shared.logging import get_logger
from .external_llm_prompt_service import ExternalLLMPromptService

LOGGER = get_logger(__name__)
OPERATION = "Tag SEGS w/ External LLM"


@dataclass(frozen=True)
class LLMTagFormattingControls:
    """Store formatting controls for external LLM tag responses."""

    replace_underscore: bool = True
    trailing_comma: bool = False
    exclude_tags: str = ""


class ExternalLLMGenerationBoundary(Protocol):
    """External LLM provider execution boundary."""

    def model_choices(self) -> list[str]:
        """Return cached provider model choices for Comfy dropdowns."""

    def generate_with_image_data_url(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
        reasoning_effort: str = DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
        image_data_url: str | None = None,
    ) -> str:
        """Return one assistant response for a pre-encoded image."""


class SegmentImageEncodingBoundary(Protocol):
    """Encode ordered SEG crops as LLM vision images."""

    def encode_segment_as_data_url(
        self,
        image: torch.Tensor,
        segment: Segment,
        mode: str,
    ) -> str:
        """Return one SEG crop image data URL."""


class ConditioningEncodingBoundary(Protocol):
    """Encode ordered prompts into a conditioning batch."""

    def encode_batch(self, clip: Any, chunks: tuple[str, ...]) -> ConditioningBatch:
        """Return conditioning entries in prompt order."""


@dataclass(frozen=True)
class TagSEGSWithExternalLLMResult:
    """Return unchanged SEGS and aligned external-LLM conditioning."""

    segs: ImpactSegs
    positive: ConditioningBatch


class TagSEGSWithExternalLLMService:
    """Caption provided SEGS crops with an external LLM and encode conditioning."""

    def __init__(
        self,
        llm: ExternalLLMGenerationBoundary | None = None,
        image_encoder: SegmentImageEncodingBoundary | None = None,
        conditioning_encoder: ConditioningEncodingBoundary | None = None,
        progress_factory: Callable[[int], ProgressReporter] | None = None,
    ) -> None:
        """Create the service with injectable runtime boundaries."""

        self._llm = llm or ExternalLLMPromptService()
        self._image_encoder = image_encoder or ExternalLLMSegsImageEncoder()
        self._conditioning_encoder = conditioning_encoder or ComfyConditioningEncoder()
        self._progress_factory = progress_factory or create_comfy_progress

    def model_choices(self) -> list[str]:
        """Return cached provider model choices for Comfy dropdowns."""

        return self._llm.model_choices()

    def tag(
        self,
        image: object,
        segs: object,
        clip: Any,
        model: str,
        system_prompt: str,
        user_prompt: str,
        universal_positive: str,
        seg_image_mode: str,
        formatting: LLMTagFormattingControls,
        max_tokens: int = DEFAULT_EXTERNAL_LLM_MAX_TOKENS,
        reasoning_effort: str = DEFAULT_EXTERNAL_LLM_REASONING_EFFORT,
    ) -> TagSEGSWithExternalLLMResult:
        """Return original SEGS plus LLM-derived conditioning in segment order."""

        image_tensor = validate_single_image(image, OPERATION)
        native_segs = coerce_segs(segs)
        self._validate_segs_target_image(native_segs, image_tensor)
        _header, segments = native_segs
        if not segments:
            raise ValueError("No SEGS were provided for Tag SEGS w/ External LLM.")

        progress = self._progress_factory(len(segments) + 2)
        progress.update(1)
        prompts: list[str] = []
        for segment in segments:
            prompts.append(
                self._prompt_for_segment(
                    image=image_tensor,
                    segment=segment,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    universal_positive=universal_positive,
                    seg_image_mode=seg_image_mode,
                    formatting=formatting,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                )
            )
            progress.update(1)

        positive = self._conditioning_encoder.encode_batch(clip, tuple(prompts))
        progress.update(1)
        if len(positive.entries) != len(segments):
            raise ValueError(
                "Conditioning encoder returned "
                f"{len(positive.entries)} entries for {len(segments)} SEGS."
            )

        LOGGER.info(
            "Tag SEGS w/ External LLM pass completed",
            extra={
                "operation": "tag_segs_with_external_llm",
                "segment_count": len(segments),
                "external_llm_model": model,
                "seg_image_mode": seg_image_mode,
                "universal_positive_present": bool(universal_positive.strip()),
                "replace_underscore": formatting.replace_underscore,
                "trailing_comma": formatting.trailing_comma,
                "exclude_tags_present": bool(formatting.exclude_tags.strip()),
            },
        )
        return TagSEGSWithExternalLLMResult(
            segs=to_impact_compatible_segs(native_segs),
            positive=positive,
        )

    def _prompt_for_segment(
        self,
        image: torch.Tensor,
        segment: Segment,
        model: str,
        system_prompt: str,
        user_prompt: str,
        universal_positive: str,
        seg_image_mode: str,
        formatting: LLMTagFormattingControls,
        max_tokens: int,
        reasoning_effort: str,
    ) -> str:
        """Generate, format, and prefix one segment prompt."""

        image_data_url = self._image_encoder.encode_segment_as_data_url(
            image,
            segment,
            seg_image_mode,
        )
        response = self._llm.generate_with_image_data_url(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            image_data_url=image_data_url,
        )
        prompt = format_external_llm_tags(response, formatting)
        return prefix_prompt(universal_positive, prompt)

    def _validate_segs_target_image(
        self,
        segs: NativeSegs,
        image: torch.Tensor,
    ) -> None:
        """Reject SEGS that cannot be cropped from the provided image."""

        header, segments = segs
        image_height = int(image.shape[1])
        image_width = int(image.shape[2])
        if header != (image_height, image_width):
            raise ValueError(
                f"{OPERATION} requires SEGS header dimensions to match the image: "
                f"SEGS is {header[0]}x{header[1]}, image is "
                f"{image_height}x{image_width}."
            )
        for index, segment in enumerate(segments):
            _validate_segment_crop(segment, index, image_height, image_width)


def format_external_llm_tags(
    response: str,
    controls: LLMTagFormattingControls,
) -> str:
    """Format one external LLM response into comma-separated prompt tags."""

    stripped = response.strip()
    if not stripped:
        raise ValueError("External LLM returned an empty response for a SEG.")

    excluded = _excluded_tags(controls)
    tags: list[str] = []
    for chunk in stripped.split(","):
        tag = _normalize_tag(chunk, controls.replace_underscore)
        if not tag or tag.lower() in excluded:
            continue
        tags.append(tag)
    if not tags:
        raise ValueError("External LLM response had no usable tags after exclusions.")

    prompt = ", ".join(tags)
    if controls.trailing_comma and not prompt.endswith(","):
        return f"{prompt},"
    return prompt


def _excluded_tags(controls: LLMTagFormattingControls) -> set[str]:
    """Return normalized excluded tag names."""

    return {
        tag.lower()
        for raw_tag in controls.exclude_tags.split(",")
        if (tag := _normalize_tag(raw_tag, controls.replace_underscore))
    }


def _normalize_tag(value: str, replace_underscore: bool) -> str:
    """Return the prompt-facing form of one tag-like response chunk."""

    tag = value.strip()
    if replace_underscore:
        return tag.replace("_", " ")
    return tag


def _validate_segment_crop(
    segment: Segment,
    index: int,
    image_height: int,
    image_width: int,
) -> None:
    """Reject a segment crop region that falls outside the image."""

    region = segment.crop_region
    if region.right <= image_width and region.bottom <= image_height:
        return
    raise ValueError(
        f"{OPERATION} SEG {index} ('{segment.label}') crop_region must fit "
        f"inside the image; got ({region.left}, {region.top}, {region.right}, "
        f"{region.bottom}) for image {image_height}x{image_width}."
    )
