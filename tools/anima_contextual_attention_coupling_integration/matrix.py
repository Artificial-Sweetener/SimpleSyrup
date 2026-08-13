# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Define the immutable P7.8 Contextual Attention Coupling matrix."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_prompts import (
    PINNED_ADAPTER_B,
    PINNED_ADAPTER_A,
    GlobalLora,
    RegionalLora,
)
from tools.attention_coupling_benchmark.manifest_types import (
    BenchmarkCase,
    MaskRectangle,
)

PUBLIC_NODE_ID = "SimpleSyrup.KSamplerAttentionCouplingContextual"
SOURCE_WIDTH = 1024
SOURCE_HEIGHT = 1024
TARGET_WIDTH = 1536
TARGET_HEIGHT = 1536
REFINEMENT_DENOISE = 0.3
CONTEXT_SIZE = 96
CONTEXT_OVERLAP = 32
CONTEXT_BATCH_SIZE = 4
GLOBAL_WEIGHT = 1.0
GLOBAL_STEPS = 3
GLOBAL_DECAY = 0.5
SUBTOKEN_MASK_CASE_ID = "sub-token-left-pixel"


@dataclass(frozen=True, slots=True)
class ContextualIntegrationCase:
    """Describe one complete Contextual public-node image and evidence case."""

    case_id: str
    label: str
    diffusion_mode: str
    branch_mode: str
    mask_case_id: str
    use_segs: bool
    global_steps: int
    cfg: float = 1.0
    feather: int = 0
    global_loras: tuple[GlobalLora, ...] = ()
    regional_loras: tuple[RegionalLora, ...] = ()

    @property
    def regional_lora_count(self) -> int:
        """Return the declared regional adapter execution count."""

        return len(self.regional_loras)


def cases() -> tuple[ContextualIntegrationCase, ...]:
    """Return local, reduced-global, and sub-token cases for both fusion modes."""

    scheduled = (
        RegionalLora(0, PINNED_ADAPTER_A, 0.8, (0.0, 0.75)),
        RegionalLora(1, PINNED_ADAPTER_B, 0.8, (0.25, 1.0)),
    )
    definitions: list[ContextualIntegrationCase] = []
    for diffusion_mode in ("multidiffusion", "mixture_of_diffusers"):
        mode_label = (
            "MultiDiffusion"
            if diffusion_mode == "multidiffusion"
            else "Mixture of Diffusers"
        )
        definitions.extend(
            (
                ContextualIntegrationCase(
                    case_id=f"{diffusion_mode}-local-segs",
                    label=(
                        f"{mode_label}; local Contextual branch; SEGS plus "
                        "overlapping regions; scheduled ADAPTER_A/ADAPTER_B"
                    ),
                    diffusion_mode=diffusion_mode,
                    branch_mode="local",
                    mask_case_id="overlapping-regions",
                    use_segs=True,
                    global_steps=0,
                    regional_loras=scheduled,
                ),
                ContextualIntegrationCase(
                    case_id=f"{diffusion_mode}-reduced-global-segs",
                    label=(
                        f"{mode_label}; local plus reduced-global correction; SEGS "
                        "plus overlapping regions; scheduled ADAPTER_A/ADAPTER_B"
                    ),
                    diffusion_mode=diffusion_mode,
                    branch_mode="reduced_global",
                    mask_case_id="overlapping-regions",
                    use_segs=True,
                    global_steps=GLOBAL_STEPS,
                    regional_loras=scheduled,
                ),
                ContextualIntegrationCase(
                    case_id=f"{diffusion_mode}-sub-token-global",
                    label=(
                        f"{mode_label}; reduced-global correction; one-pixel "
                        "sub-token region; scheduled ADAPTER_A/ADAPTER_B"
                    ),
                    diffusion_mode=diffusion_mode,
                    branch_mode="sub_token_reduced_global",
                    mask_case_id=SUBTOKEN_MASK_CASE_ID,
                    use_segs=False,
                    global_steps=GLOBAL_STEPS,
                    regional_loras=scheduled,
                ),
            )
        )
    return tuple(definitions)


def select_cases(case_ids: tuple[str, ...]) -> tuple[ContextualIntegrationCase, ...]:
    """Return an ordered diagnostic subset or the complete matrix."""

    definitions = cases()
    if not case_ids:
        return definitions
    definitions_by_id = {case.case_id: case for case in definitions}
    unknown = tuple(case_id for case_id in case_ids if case_id not in definitions_by_id)
    if unknown:
        raise ValueError(f"Unknown P7.8 case ids: {unknown!r}.")
    return tuple(definitions_by_id[case_id] for case_id in case_ids)


def subtoken_mask_case() -> BenchmarkCase:
    """Return two masks whose first region is exactly one image pixel wide."""

    one_pixel = 1.0 / TARGET_WIDTH
    return BenchmarkCase(
        case_id=SUBTOKEN_MASK_CASE_ID,
        scenario_tags=("sub_token_region",),
        global_prompt="",
        regional_prompts=("", ""),
        masks=(
            MaskRectangle(0.0, 0.0, one_pixel, 1.0),
            MaskRectangle(one_pixel, 0.0, 1.0, 1.0),
        ),
        regional_prompt_weight=1.0,
        region_mask_feather=0,
    )
