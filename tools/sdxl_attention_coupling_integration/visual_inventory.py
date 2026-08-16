# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load external model artifacts and prompts for managed visual evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tools.comfy_api import JsonObject


@dataclass(frozen=True, slots=True)
class VisualCharacterInventory:
    """Describe one externally selected character adapter and its trigger prompts."""

    source: Path
    label: str
    prompt_g: str
    prompt_l: str

    def __post_init__(self) -> None:
        """Require one existing artifact and complete display and prompt values."""

        _validate_source(self.source, role="character adapter")
        _validate_text(self.label, role="character label")
        _validate_text(self.prompt_g, role="character G prompt")
        _validate_text(self.prompt_l, role="character L prompt")


@dataclass(frozen=True, slots=True)
class VisualAdapterInventory:
    """Describe one externally selected style adapter and its trigger prompts."""

    source: Path
    label: str
    prompt_g: str
    prompt_l: str

    def __post_init__(self) -> None:
        """Require one existing adapter, label, and complete trigger prompts."""

        _validate_source(self.source, role="adapter")
        _validate_text(self.label, role="adapter label")
        _validate_text(self.prompt_g, role="adapter G prompt")
        _validate_text(self.prompt_l, role="adapter L prompt")


@dataclass(frozen=True, slots=True)
class SdxlVisualInventory:
    """Provide all external artifacts required by the generic visual matrix."""

    checkpoint_source: Path
    checkpoint_label: str
    left_character: VisualCharacterInventory
    right_character: VisualCharacterInventory
    style: VisualAdapterInventory

    def __post_init__(self) -> None:
        """Require a complete checkpoint and three distinct adapter files."""

        _validate_source(self.checkpoint_source, role="checkpoint")
        _validate_text(self.checkpoint_label, role="checkpoint label")
        for role, adapter in (
            ("left character adapter", self.left_character),
            ("right character adapter", self.right_character),
            ("style adapter", self.style),
        ):
            _validate_source(adapter.source, role=role)
        adapter_sources = {
            self.left_character.source.resolve(),
            self.right_character.source.resolve(),
            self.style.source.resolve(),
        }
        if len(adapter_sources) != 3:
            raise ValueError("Visual inventory adapter sources must be distinct.")

    @classmethod
    def load(cls, path: Path) -> SdxlVisualInventory:
        """Decode one external JSON inventory without retaining its path."""

        try:
            decoded: object = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Visual inventory must contain valid JSON.") from error
        if not isinstance(decoded, dict):
            raise TypeError("Visual inventory must be a JSON object.")
        payload = cast(JsonObject, decoded)
        return cls(
            checkpoint_source=_path(payload, "checkpoint_source"),
            checkpoint_label=_text(payload, "checkpoint_label"),
            left_character=_character(payload, "left_character"),
            right_character=_character(payload, "right_character"),
            style=_adapter(payload, "style"),
        )


def _character(payload: JsonObject, key: str) -> VisualCharacterInventory:
    """Decode one required character role object."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Visual inventory {key!r} must be an object.")
    narrowed = cast(JsonObject, value)
    return VisualCharacterInventory(
        source=_path(narrowed, "source"),
        label=_text(narrowed, "label"),
        prompt_g=_text(narrowed, "prompt_g"),
        prompt_l=_text(narrowed, "prompt_l"),
    )


def _adapter(payload: JsonObject, key: str) -> VisualAdapterInventory:
    """Decode one required adapter role object."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Visual inventory {key!r} must be an object.")
    narrowed = cast(JsonObject, value)
    return VisualAdapterInventory(
        source=_path(narrowed, "source"),
        label=_text(narrowed, "label"),
        prompt_g=_text(narrowed, "prompt_g"),
        prompt_l=_text(narrowed, "prompt_l"),
    )


def _path(payload: JsonObject, key: str) -> Path:
    """Return one required absolute inventory path."""

    value = _text(payload, key)
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Visual inventory {key!r} must be an absolute path.")
    return path


def _text(payload: JsonObject, key: str) -> str:
    """Return one required non-empty inventory string."""

    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Visual inventory {key!r} must be non-empty text.")
    return value.strip()


def _validate_source(path: Path, *, role: str) -> None:
    """Require one absolute existing safetensors artifact."""

    if not isinstance(path, Path) or not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"Visual {role} source does not exist: {path!s}")
    if path.suffix.casefold() != ".safetensors":
        raise ValueError(f"Visual {role} must be a safetensors artifact.")


def _validate_text(value: str, *, role: str) -> None:
    """Require one non-empty external label or prompt."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Visual {role} must be non-empty text.")
