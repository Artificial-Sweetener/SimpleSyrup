# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own exact Anima regional-LoRA target classification and admission."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum

from .anima_target_names import (
    ANIMA_LORA_TARGET_NAME_NORMALIZER,
    AnimaLoraTargetNameNormalizer,
)
from .anima_target_ownership import (
    AnimaRegionalLoraTargetOwner,
    classify_anima_regional_lora_target_owner,
)
from .standard_adapter import (
    STANDARD_LORA_ADAPTER_DECODER,
    StandardLoraAdapterDecoder,
    StandardLoraTarget,
)

_TARGET_PATTERN = re.compile(r"^diffusion_model\.blocks\.(\d+)\.(.+)$")
ANIMA_BLOCK_COUNT = 28


class AnimaLoraTargetFamily(StrEnum):
    """Name every Anima linear family with implemented regional semantics."""

    SELF_ATTN_Q = "self_attn.q_proj"
    SELF_ATTN_K = "self_attn.k_proj"
    SELF_ATTN_V = "self_attn.v_proj"
    SELF_ATTN_OUTPUT = "self_attn.output_proj"
    CROSS_ATTN_Q = "cross_attn.q_proj"
    CROSS_ATTN_K = "cross_attn.k_proj"
    CROSS_ATTN_V = "cross_attn.v_proj"
    CROSS_ATTN_OUTPUT = "cross_attn.output_proj"
    MLP_LAYER1 = "mlp.layer1"
    MLP_LAYER2 = "mlp.layer2"
    ADALN_SELF_ATTN_1 = "adaln_modulation_self_attn.1"
    ADALN_SELF_ATTN_2 = "adaln_modulation_self_attn.2"
    ADALN_CROSS_ATTN_1 = "adaln_modulation_cross_attn.1"
    ADALN_CROSS_ATTN_2 = "adaln_modulation_cross_attn.2"
    ADALN_MLP_1 = "adaln_modulation_mlp.1"
    ADALN_MLP_2 = "adaln_modulation_mlp.2"


_EXPECTED_FEATURES: dict[AnimaLoraTargetFamily, tuple[int, int]] = {
    AnimaLoraTargetFamily.SELF_ATTN_Q: (2048, 2048),
    AnimaLoraTargetFamily.SELF_ATTN_K: (2048, 2048),
    AnimaLoraTargetFamily.SELF_ATTN_V: (2048, 2048),
    AnimaLoraTargetFamily.SELF_ATTN_OUTPUT: (2048, 2048),
    AnimaLoraTargetFamily.CROSS_ATTN_Q: (2048, 2048),
    AnimaLoraTargetFamily.CROSS_ATTN_K: (1024, 2048),
    AnimaLoraTargetFamily.CROSS_ATTN_V: (1024, 2048),
    AnimaLoraTargetFamily.CROSS_ATTN_OUTPUT: (2048, 2048),
    AnimaLoraTargetFamily.MLP_LAYER1: (2048, 8192),
    AnimaLoraTargetFamily.MLP_LAYER2: (8192, 2048),
    AnimaLoraTargetFamily.ADALN_SELF_ATTN_1: (2048, 256),
    AnimaLoraTargetFamily.ADALN_SELF_ATTN_2: (256, 6144),
    AnimaLoraTargetFamily.ADALN_CROSS_ATTN_1: (2048, 256),
    AnimaLoraTargetFamily.ADALN_CROSS_ATTN_2: (256, 6144),
    AnimaLoraTargetFamily.ADALN_MLP_1: (2048, 256),
    AnimaLoraTargetFamily.ADALN_MLP_2: (256, 6144),
}


def anima_lora_target_name(
    block_index: int,
    family: AnimaLoraTargetFamily,
) -> str:
    """Build one canonical admitted Anima diffusion target name."""

    if not 0 <= block_index < ANIMA_BLOCK_COUNT:
        raise ValueError("Anima block index must be between 0 and 27")
    return f"diffusion_model.blocks.{block_index}.{family.value}"


def expected_anima_lora_features(
    family: AnimaLoraTargetFamily,
) -> tuple[int, int]:
    """Return the authoritative input and output features for one family."""

    return _EXPECTED_FEATURES[family]


@dataclass(frozen=True)
class AnimaLoraAdmissionIssue:
    """Describe one complete pre-device adapter admission failure."""

    key: str
    reason: str


class AnimaLoraAdmissionError(ValueError):
    """Report every unsupported format and target from one adapter."""

    def __init__(self, issues: tuple[AnimaLoraAdmissionIssue, ...]) -> None:
        """Create one actionable aggregate diagnostic."""

        self.issues = issues
        details = "\n".join(f"- {issue.key}: {issue.reason}" for issue in issues)
        super().__init__(
            "Regional Anima LoRA admission failed before device work:\n" + details
        )


@dataclass(frozen=True)
class AnimaLoraTarget:
    """Bind one decoded target to an exact implemented Anima block family."""

    block_index: int
    family: AnimaLoraTargetFamily
    adapter: StandardLoraTarget


@dataclass(frozen=True)
class AnimaLoraAdmission:
    """Retain every classified target only after complete adapter admission."""

    targets: tuple[AnimaLoraTarget, ...]


class AnimaLoraTargetClassifier:
    """Admit only complete standard adapters mapped to implemented Anima targets."""

    def __init__(
        self,
        decoder: StandardLoraAdapterDecoder = STANDARD_LORA_ADAPTER_DECODER,
        target_names: AnimaLoraTargetNameNormalizer = (
            ANIMA_LORA_TARGET_NAME_NORMALIZER
        ),
    ) -> None:
        """Use independent owners for adapter decoding and Anima target names."""

        self._decoder = decoder
        self._target_names = target_names

    def admit(self, weights: object) -> AnimaLoraAdmission:
        """Collect all format, target, and dimension failures before rejection."""

        decoded = self._decoder.decode(weights)
        issues = [
            AnimaLoraAdmissionIssue(issue.key, issue.reason) for issue in decoded.issues
        ]
        targets: list[AnimaLoraTarget] = []
        for adapter in decoded.targets:
            classified, target_issues = self._classify(adapter)
            issues.extend(target_issues)
            if classified is not None:
                targets.append(classified)
        if not issues and not targets:
            issues.append(
                AnimaLoraAdmissionIssue(
                    "<adapter>",
                    "adapter must contain at least one complete supported target",
                )
            )
        if issues:
            raise AnimaLoraAdmissionError(tuple(issues))
        return AnimaLoraAdmission(tuple(targets))

    def _classify(
        self,
        adapter: StandardLoraTarget,
    ) -> tuple[AnimaLoraTarget | None, tuple[AnimaLoraAdmissionIssue, ...]]:
        """Map one decoded target to an exact block, family, and feature shape."""

        canonical_target = self._target_names.normalize(adapter.target)
        if canonical_target != adapter.target:
            adapter = replace(adapter, target=canonical_target)
        ownership = classify_anima_regional_lora_target_owner(adapter.target)
        if ownership.owner is not AnimaRegionalLoraTargetOwner.IMAGE_TRANSFORMER_BLOCK:
            assert ownership.rejection_reason is not None
            return None, (
                AnimaLoraAdmissionIssue(
                    adapter.target,
                    ownership.rejection_reason,
                ),
            )
        match = _TARGET_PATTERN.fullmatch(adapter.target)
        if match is None:
            return None, (
                AnimaLoraAdmissionIssue(
                    adapter.target,
                    "unsupported Anima target path",
                ),
            )
        block_index = int(match.group(1))
        family_name = match.group(2)
        if not 0 <= block_index < ANIMA_BLOCK_COUNT:
            return None, (
                AnimaLoraAdmissionIssue(
                    adapter.target,
                    "unsupported Anima block index; expected 0 through 27",
                ),
            )
        try:
            family = AnimaLoraTargetFamily(family_name)
        except ValueError:
            return None, (
                AnimaLoraAdmissionIssue(
                    adapter.target,
                    "unsupported Anima target family",
                ),
            )
        expected_input, expected_output = expected_anima_lora_features(family)
        if (
            adapter.input_features != expected_input
            or adapter.output_features != expected_output
        ):
            return None, (
                AnimaLoraAdmissionIssue(
                    adapter.target,
                    "unsupported Anima target shape; expected "
                    f"input={expected_input}, output={expected_output}, observed "
                    f"input={adapter.input_features}, output={adapter.output_features}",
                ),
            )
        return AnimaLoraTarget(block_index, family, adapter), ()


ANIMA_LORA_TARGET_CLASSIFIER = AnimaLoraTargetClassifier()
