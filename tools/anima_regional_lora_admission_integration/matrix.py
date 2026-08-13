# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the closed P9.5 regional diffusion-WeightHook admission matrix."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_prompts import PINNED_ADAPTER_A

from .graph_contract import (
    CFG,
    PublicRegionalLoraHook,
    RegionalLoraAdmissionGraphCase,
)

PINNED_ADAPTER_A_IDENTITY = "supported-adapter_a"
PINNED_ADAPTER_A_HOOK = PublicRegionalLoraHook(
    PINNED_ADAPTER_A,
    0.75,
    PINNED_ADAPTER_A_IDENTITY,
)


@dataclass(frozen=True, slots=True)
class RegionalLoraAdmissionCase(RegionalLoraAdmissionGraphCase):
    """Describe one exact supported or fail-closed managed workflow."""

    expected_exception_suffix: str | None
    expected_error_fragments: tuple[str, ...] = ()

    @property
    def expect_success(self) -> bool:
        """Return whether this case must render a supported image."""

        return self.expected_exception_suffix is None


def cases() -> tuple[RegionalLoraAdmissionCase, ...]:
    """Return the exact success, classification, and atomicity matrix."""

    non_weight_hook = "non-weight-hook"
    model_as_lora_format = "model-as-lora-format"
    unsupported_suffix_format = "unsupported-suffix-format"
    incomplete_standard_pair = "incomplete-standard-pair"
    unsupported_anima_target = "unsupported-anima-target"
    return (
        RegionalLoraAdmissionCase(
            "supported-adapter_a",
            "Supported full-surface regional ADAPTER_A WeightHook",
            (PINNED_ADAPTER_A_HOOK,),
            None,
            None,
            CFG,
            0,
            None,
        ),
        RegionalLoraAdmissionCase(
            non_weight_hook,
            "Reject non-WeightHook regional transformer options",
            (),
            non_weight_hook,
            None,
            CFG,
            0,
            "TypeError",
            ("unsupported hooks", "TransformerOptionsHook"),
        ),
        RegionalLoraAdmissionCase(
            model_as_lora_format,
            "Reject converted model-as-LoRA regional payload",
            (),
            model_as_lora_format,
            model_as_lora_format,
            CFG,
            0,
            "AnimaRegionalLoraPlanAdmissionError",
            ("failed before sampling", "unsupported adapter format"),
        ),
        RegionalLoraAdmissionCase(
            unsupported_suffix_format,
            "Reject unsupported lora_down/lora_up suffixes",
            (),
            unsupported_suffix_format,
            unsupported_suffix_format,
            CFG,
            0,
            "AnimaRegionalLoraPlanAdmissionError",
            (
                "failed before sampling",
                "lora_down.weight",
                "lora_up.weight",
                "unsupported adapter format",
            ),
        ),
        RegionalLoraAdmissionCase(
            incomplete_standard_pair,
            "Reject incomplete standard regional adapter pair",
            (),
            incomplete_standard_pair,
            incomplete_standard_pair,
            CFG,
            0,
            "AnimaRegionalLoraPlanAdmissionError",
            ("failed before sampling", "incomplete adapter pair", "missing B"),
        ),
        RegionalLoraAdmissionCase(
            unsupported_anima_target,
            "Reject unsupported Anima diffusion target family",
            (),
            unsupported_anima_target,
            unsupported_anima_target,
            CFG,
            0,
            "AnimaRegionalLoraPlanAdmissionError",
            (
                "failed before sampling",
                "unsupported Anima target family",
                "unknown_projection",
            ),
        ),
        RegionalLoraAdmissionCase(
            "mixed-supported-unsupported-target",
            "Reject complete ADAPTER_A plus unsupported target atomically",
            (PINNED_ADAPTER_A_HOOK,),
            unsupported_anima_target,
            unsupported_anima_target,
            CFG,
            0,
            "AnimaRegionalLoraPlanAdmissionError",
            (
                "failed before sampling",
                "adapter 1",
                unsupported_anima_target,
                "unsupported Anima target family",
            ),
        ),
    )
