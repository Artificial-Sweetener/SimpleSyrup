# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own installed real-model fixtures used by regional LoRA characterization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import comfy.sd

from simple_syrup.domain.regional_lora_plan import (
    RegionalLoraAdapterIdentity,
    RegionalLoraAdapterPlan,
    RegionalLoraBranch,
    RegionalLoraPlan,
    RegionalLoraScheduleBoundary,
)
from tools.comfy_api import JsonObject


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


def load_real_fixtures(path: Path) -> dict[str, RegionalLoraRealFixture]:
    """Load externally selected model/adapter pairs keyed by capability role."""

    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            "Regional real-fixture inventory must be valid JSON."
        ) from error
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError("Regional real-fixture inventory must be a non-empty object.")
    fixtures: dict[str, RegionalLoraRealFixture] = {}
    for role, raw in decoded.items():
        if not isinstance(role, str) or not role or not isinstance(raw, dict):
            raise TypeError("Regional real-fixture entries must be named objects.")
        entry = cast(JsonObject, raw)
        model_path = _absolute_path(entry, "model_path")
        lora_path = _absolute_path(entry, "lora_path")
        standalone = entry.get("standalone_diffusion_model")
        if not isinstance(standalone, bool):
            raise TypeError(
                "Regional fixture standalone_diffusion_model must be boolean."
            )
        fixture = RegionalLoraRealFixture(
            family=role,
            model_path=model_path,
            lora_path=lora_path,
            standalone_diffusion_model=standalone,
        )
        fixture.require_files()
        fixtures[role] = fixture
    return fixtures


def _absolute_path(payload: JsonObject, key: str) -> Path:
    """Return one required absolute external artifact path."""

    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Regional fixture {key!r} must be non-empty text.")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Regional fixture {key!r} must be absolute.")
    return path
