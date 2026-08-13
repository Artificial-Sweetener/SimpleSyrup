# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own the immutable pinned ADAPTER_A global characterization matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LoraMode = Literal["static", "scheduled"]

PINNED_LORA_NAME = "Anima\\style\\adapter-a.safetensors"
PINNED_LORA_SIZE = 138_663_768
PINNED_LORA_SHA256 = "0c915b59f464fd3d72c49a241440f8a41582ec76b597b0029e18b657b600c7d8"
MEASUREMENT_SEEDS = (1_029_384_756, 3_141_592_653, 2_718_281_828)
CAPTURE_SEED = MEASUREMENT_SEEDS[0]
SCHEDULE = ((0.0, 0.0), (0.25, 1.0), (0.5, 0.5), (0.75, 0.0))


@dataclass(frozen=True)
class AdapterDefinition:
    """Define one ordered identity applied from the pinned adapter artifact."""

    identity: str
    strength: float
    schedule: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class LoraProfile:
    """Define one zero, static, or scheduled adapter-stack profile."""

    profile_id: str
    mode: LoraMode
    adapters: tuple[AdapterDefinition, ...]


@dataclass(frozen=True)
class LoraRun:
    """Identify one measurement or denoiser-digest capture execution."""

    profile: LoraProfile
    seed: int
    capture_outputs: bool

    @property
    def artifact_id(self) -> str:
        """Return the stable result and image identity."""

        evidence_kind = "outputs" if self.capture_outputs else "measurement"
        return (
            f"adapter_a-global__{self.profile.profile_id}__{evidence_kind}__seed-{self.seed}"
        )


def profiles() -> tuple[LoraProfile, ...]:
    """Return the fixed additive-strength and scheduling profiles."""

    static_strengths = {
        0: (),
        1: (1.0,),
        2: (0.4, 0.6),
        4: (0.1, 0.2, 0.3, 0.4),
    }
    static = tuple(
        LoraProfile(
            profile_id=f"static-{count}",
            mode="static",
            adapters=tuple(
                AdapterDefinition(f"static-{count}-adapter-{index + 1}", strength)
                for index, strength in enumerate(strengths)
            ),
        )
        for count, strengths in static_strengths.items()
    )
    scheduled = tuple(
        LoraProfile(
            profile_id=f"scheduled-{count}",
            mode="scheduled",
            adapters=tuple(
                AdapterDefinition(
                    f"scheduled-{count}-adapter-{index + 1}", strength, SCHEDULE
                )
                for index, strength in enumerate(strengths)
            ),
        )
        for count, strengths in static_strengths.items()
        if count > 0
    )
    return (*static, *scheduled)


def runs() -> tuple[LoraRun, ...]:
    """Expand three clean timing seeds plus one output-evidence pass per profile."""

    expanded: list[LoraRun] = []
    for profile in profiles():
        expanded.extend(
            LoraRun(profile, seed, capture_outputs=False) for seed in MEASUREMENT_SEEDS
        )
        expanded.append(LoraRun(profile, CAPTURE_SEED, capture_outputs=True))
    return tuple(expanded)
