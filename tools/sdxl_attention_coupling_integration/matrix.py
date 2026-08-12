"""Define the fixed Phase 8 SDXL managed-workflow matrix."""

from __future__ import annotations

from dataclasses import dataclass

CHECKPOINT_STABLE_NAME = "juggernautXL_juggXIByRundiffusion.safetensors"
CHECKPOINT_SOURCE_NAME = f"SDXL\\{CHECKPOINT_STABLE_NAME}"
CHECKPOINT_SIZE = 7_105_350_536
CHECKPOINT_SHA256 = "33e58e86686f6b386c526682b5da9228ead4f91d994abd4b053442dc5b42719e"
SOURCE_WIDTH = 1024
SOURCE_HEIGHT = 1024
TARGET_WIDTH = 1536
TARGET_HEIGHT = 1536
UPSCALE_FACTOR = 1.5
REGION_WIDTH = round(TARGET_WIDTH * 0.4)
RIGHT_REGION_START = TARGET_WIDTH - REGION_WIDTH
SEED = 7_429_113_057
CFG = 5.0
SAMPLER = "dpmpp_2m_sde"
SCHEDULER = "karras"
SOURCE_STEPS = 24
REFINEMENT_STEPS = 12
REFINEMENT_DENOISE = 0.25
TILE_SIZE = 128
TILE_OVERLAP = 64
TILE_BATCH_SIZE = 4
CONTEXTUAL_EXPECTED_MODEL_CALLS = REFINEMENT_STEPS * 5 + 1
POSITIVE_PROMPTS = (
    "cinematic wide full-body environmental scene of exactly two separate adult "
    "adventurers, one person at the far left and one person at the far right, "
    "large empty forest clearing between them, both complete bodies visible, "
    "clearly separated people, coherent perspective",
    "one complete distinct silver-haired male knight in blue armor at the far "
    "left, full body, standing alone",
    "one complete distinct red-haired female mage in crimson robes at the far "
    "right, full body, standing alone",
)
NEGATIVE_PROMPTS = (
    "single person, centered person, touching people, overlapping people, fused "
    "people, merged body, split face, half-and-half person, duplicate body, "
    "collage, split screen, cropped, low quality, blurry",
    "low quality, malformed armor, fused body",
    "low quality, malformed hands, fused body",
)


@dataclass(frozen=True, slots=True)
class SdxlIntegrationMode:
    """Describe one exact public-node execution and acceptance geometry."""

    mode_id: str
    label: str
    node_id: str
    width: int
    height: int
    expected_model_calls: int
    expected_spatial_modes: frozenset[str]


MODES = (
    SdxlIntegrationMode(
        "full",
        "SDXL full 1024 Attention Coupling",
        "SimpleSyrup.KSamplerAttentionCoupling",
        SOURCE_WIDTH,
        SOURCE_HEIGHT,
        SOURCE_STEPS,
        frozenset({"full"}),
    ),
    SdxlIntegrationMode(
        "tiled-1.5x",
        "SDXL tiled 1024 to 1536 Attention Coupling",
        "SimpleSyrup.KSamplerAttentionCouplingTiled",
        TARGET_WIDTH,
        TARGET_HEIGHT,
        REFINEMENT_STEPS,
        frozenset({"tile"}),
    ),
    SdxlIntegrationMode(
        "contextual-1.5x",
        "SDXL Contextual 1024 to 1536 Attention Coupling",
        "SimpleSyrup.KSamplerAttentionCouplingContextual",
        TARGET_WIDTH,
        TARGET_HEIGHT,
        CONTEXTUAL_EXPECTED_MODEL_CALLS,
        frozenset({"tile", "contextual_global"}),
    ),
)
