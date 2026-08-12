"""Define the shared fixed regional-LoRA admission graph contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

PUBLIC_NODE_ID = "SimpleSyrup.KSamplerAttentionCoupling"
MASK_CASE_ID = "vertical-hard-50-50"
WIDTH = 512
HEIGHT = 512
STEPS = 8
CFG = 4.0
BASE_PROMPT = (
    "masterpiece, best quality, score_7, safe, two distinct people standing "
    "side by side, full body, balanced city-street composition"
)
REGION_ZERO_PROMPT = "one person on the left wearing a red jacket"
REGION_ONE_PROMPT = "one person on the right wearing a blue jacket"
NEGATIVE_PROMPT = "low quality, blurry, fused people, duplicate subject"


@dataclass(frozen=True, slots=True)
class PublicRegionalLoraHook:
    """Describe one public model-only LoRA hook in declared graph order."""

    lora_name: str
    strength_model: float
    adapter_identity: str

    def __post_init__(self) -> None:
        """Reject graph descriptors that cannot form a stable public hook."""

        if not self.lora_name:
            raise ValueError("Regional LoRA graph names must not be empty.")
        if not math.isfinite(self.strength_model):
            raise ValueError("Regional LoRA graph strengths must be finite.")
        if not self.adapter_identity:
            raise ValueError("Regional LoRA graph identities must not be empty.")


@dataclass(frozen=True, slots=True)
class RegionalLoraAdmissionGraphCase:
    """Describe only values consumed by the shared public admission graph."""

    case_id: str
    label: str
    public_loras: tuple[PublicRegionalLoraHook, ...]
    fixture: str | None
    fixture_adapter_identity: str | None
    cfg: float
    feather: int

    def __post_init__(self) -> None:
        """Validate stable artifact and hook-label graph inputs."""

        if not self.case_id:
            raise ValueError("Regional LoRA graph case IDs must not be empty.")
        if not self.label:
            raise ValueError("Regional LoRA graph labels must not be empty.")
        if self.fixture_adapter_identity is not None and self.fixture is None:
            raise ValueError("Fixture identities require a fixture hook source.")
        if not math.isfinite(self.cfg) or self.cfg <= 0.0:
            raise ValueError("Regional LoRA graph CFG must be finite and positive.")
        if isinstance(self.feather, bool) or not isinstance(self.feather, int):
            raise TypeError("Regional LoRA graph feather must be an integer.")
        if self.feather < 0:
            raise ValueError("Regional LoRA graph feather must not be negative.")
        identities = self.adapter_identities
        if len(set(identities)) != len(identities):
            raise ValueError("Regional LoRA graph identities must be unique.")

    @property
    def adapter_identities(self) -> tuple[str, ...]:
        """Return stable labels in exact public hook-composition order."""

        identities = tuple(adapter.adapter_identity for adapter in self.public_loras)
        if self.fixture_adapter_identity is not None:
            identities += (self.fixture_adapter_identity,)
        return identities
