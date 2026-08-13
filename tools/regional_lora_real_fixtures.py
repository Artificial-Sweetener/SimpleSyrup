# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own installed real-model fixtures used by regional LoRA characterization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import comfy.sd

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)


@dataclass(frozen=True, slots=True)
class RegionalLoraRealFixture:
    """Describe one installed model/LoRA pair and its Comfy loading boundary."""

    family: str
    model_path: Path
    lora_path: Path
    standalone_diffusion_model: bool

    def require_files(self) -> None:
        """Fail clearly when one machine-local fixture is unavailable."""

        for path in (self.model_path, self.lora_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required characterization artifact is missing: {path}"
                )

    def load_model(self) -> Any:
        """Load the fixture through its installed Comfy model boundary."""

        self.require_files()
        if self.standalone_diffusion_model:
            return comfy.sd.load_diffusion_model(str(self.model_path), model_options={})
        loaded = comfy.sd.load_checkpoint_guess_config(
            str(self.model_path),
            output_vae=False,
            output_clip=False,
            embedding_directory=[],
        )
        return loaded[0]

    def plan(self) -> RegionalLoraPlan:
        """Return one valid canonical adapter use for this fixture."""

        return RegionalLoraPlan(
            (
                RegionalLoraAdapterPlan(
                    adapter_identity=RegionalLoraAdapterIdentity(str(self.lora_path)),
                    composition_index=0,
                    region_index=0,
                    branch=RegionalLoraBranch.POSITIVE,
                    model_strength=1.0,
                    schedule=(RegionalLoraScheduleBoundary(0.0, 1.0, 1.0, 0),),
                ),
            )
        )


REGIONAL_LORA_REAL_FIXTURES = {
    "anima": RegionalLoraRealFixture(
        family="anima",
        model_path=Path(
            r"<MODEL_ROOT>\diffusion_models\Anima\diffusion-model.safetensors"
        ),
        lora_path=Path(r"<MODEL_ROOT>\Loras\Anima\style\adapter-a.safetensors"),
        standalone_diffusion_model=True,
    ),
    "sd15": RegionalLoraRealFixture(
        family="sd15",
        model_path=Path(
            r"<MODEL_ROOT>\stable-diffusion\SD 1.5\checkpoint-fixture.safetensors"
        ),
        lora_path=Path(r"<MODEL_ROOT>\lorabka\SD 1.5\adapter-fixture.safetensors"),
        standalone_diffusion_model=False,
    ),
    "sdxl": RegionalLoraRealFixture(
        family="sdxl",
        model_path=Path(
            "<MODEL_ROOT>\\stable-diffusion\\Illustrious\\"
            "checkpoint_aXLIllustrious_v30WIP.safetensors"
        ),
        lora_path=Path(
            "<MODEL_ROOT>\\Loras\\Illustrious\\Style\\"
            "ELDEN RING Background_illustriousXL_v2.safetensors"
        ),
        standalone_diffusion_model=False,
    ),
}
