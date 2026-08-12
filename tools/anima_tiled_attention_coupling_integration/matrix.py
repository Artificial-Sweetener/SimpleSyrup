"""Define the immutable P6.9 tiled Attention Coupling workflow matrix."""

from __future__ import annotations

from dataclasses import dataclass

from tools.anima_attention_coupling_prompts import (
    GLOBAL_GLOBAL_ADAPTER,
    PINNED_ADAPTER_B,
    PINNED_ADAPTER_A,
    GlobalLora,
    RegionalLora,
)

PUBLIC_NODE_ID = "SimpleSyrup.KSamplerAttentionCouplingTiled"
SOURCE_WIDTH = 1024
SOURCE_HEIGHT = 1024
TARGET_WIDTH = 1536
TARGET_HEIGHT = 1536
REFINEMENT_DENOISE = 0.3
TILE_WIDTH = 96
TILE_HEIGHT = 96
TILE_OVERLAP = 32


@dataclass(frozen=True, slots=True)
class TiledIntegrationCase:
    """Describe one complete tiled public-node image and evidence case."""

    case_id: str
    label: str
    diffusion_mode: str
    tile_batch_size: int
    mask_case_id: str
    cfg: float = 1.0
    feather: int = 0
    global_loras: tuple[GlobalLora, ...] = ()
    regional_loras: tuple[RegionalLora, ...] = ()

    @property
    def regional_lora_count(self) -> int:
        """Return the declared model-side regional adapter count."""

        return len(self.regional_loras)


def cases() -> tuple[TiledIntegrationCase, ...]:
    """Return both fusion modes at tile batches 1, 2, 4, and 8."""

    scheduled = (
        RegionalLora(0, PINNED_ADAPTER_A, 0.8, (0.0, 0.75)),
        RegionalLora(1, PINNED_ADAPTER_B, 0.8, (0.25, 1.0)),
    )
    one_adapter_a = (RegionalLora(0, PINNED_ADAPTER_A, 0.8),)
    definitions: list[TiledIntegrationCase] = []
    for diffusion_mode in ("multidiffusion", "mixture_of_diffusers"):
        mode_label = (
            "MultiDiffusion"
            if diffusion_mode == "multidiffusion"
            else "Mixture of Diffusers"
        )
        for tile_batch_size in (1, 2, 4, 8):
            scheduled_case = tile_batch_size != 4
            uncovered = tile_batch_size in {2, 8}
            definitions.append(
                TiledIntegrationCase(
                    case_id=f"{diffusion_mode}-batch-{tile_batch_size}",
                    label=(
                        f"{mode_label} upscale refinement; tile batch "
                        f"{tile_batch_size}; "
                        + (
                            "scheduled ADAPTER_A/ADAPTER_B; "
                            if scheduled_case
                            else "single ADAPTER_A; "
                        )
                        + (
                            "contained, boundary, and uncovered tiles"
                            if uncovered
                            else "contained, boundary, and overlap tiles"
                        )
                    ),
                    diffusion_mode=diffusion_mode,
                    tile_batch_size=tile_batch_size,
                    mask_case_id=(
                        "uncovered-center-strip" if uncovered else "overlapping-regions"
                    ),
                    cfg=4.0 if tile_batch_size == 8 else 1.0,
                    global_loras=(
                        (GlobalLora(GLOBAL_GLOBAL_ADAPTER, 0.35),) if tile_batch_size == 8 else ()
                    ),
                    regional_loras=scheduled if scheduled_case else one_adapter_a,
                )
            )
    return tuple(definitions)


def select_cases(case_ids: tuple[str, ...]) -> tuple[TiledIntegrationCase, ...]:
    """Return the requested ordered subset or the complete matrix when empty."""

    definitions = cases()
    if not case_ids:
        return definitions
    definitions_by_id = {case.case_id: case for case in definitions}
    unknown = tuple(case_id for case_id in case_ids if case_id not in definitions_by_id)
    if unknown:
        raise ValueError(f"Unknown P6.9 case ids: {unknown!r}.")
    return tuple(definitions_by_id[case_id] for case_id in case_ids)
