# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Loaded model containers used by SimpleSyrup masking nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_device_manager import ManagedTorchModel
    from .wd14_tagger import WD14Session, WD14TagRecord


@dataclass(frozen=True)
class LoadedSAMModel:
    """Loaded SAM-compatible model object with source metadata."""

    model: object
    source: str
    model_id: str
    managed_model: ManagedTorchModel | None = None


@dataclass(frozen=True)
class LoadedGroundingDINOModel:
    """Loaded GroundingDINO-compatible model object with text encoder metadata."""

    model: object
    text_encoder_path: Path
    source: str
    model_id: str
    managed_model: ManagedTorchModel | None = None


@dataclass(frozen=True)
class LoadedViTMatteModel:
    """Loaded ViTMatte-compatible model object with source metadata."""

    model: object
    processor: object
    source: str
    model_id: str
    model_path: Path
    managed_model: ManagedTorchModel | None = None


@dataclass(frozen=True)
class LoadedWD14Tagger:
    """Loaded WD14 tagger runtime shared by compatible nodes."""

    model_id: str
    source: str
    onnx_path: Path
    csv_path: Path
    providers: tuple[str, ...]
    session: WD14Session
    tags: tuple[WD14TagRecord, ...]


def unwrap_sam_model(model: object) -> object:
    """Return the underlying SAM object for SimpleSyrup loaded containers."""

    if isinstance(model, LoadedSAMModel):
        return model.model
    return model


def unwrap_grounding_dino_model(model: object) -> object:
    """Return the underlying GroundingDINO object for SimpleSyrup containers."""

    if isinstance(model, LoadedGroundingDINOModel):
        return model.model
    return model


def unwrap_vitmatte_model(model: object) -> LoadedViTMatteModel:
    """Return a validated SimpleSyrup ViTMatte model container."""

    if isinstance(model, LoadedViTMatteModel):
        return model
    raise TypeError(
        "VITMATTE_MODEL is not compatible with Prompt SEGS w/ SAM. Expected a "
        "ViTMatte model loaded by ViTMatte Model Loader."
    )


def unwrap_wd14_tagger(model: object) -> LoadedWD14Tagger:
    """Return a validated SimpleSyrup WD14 tagger container."""

    if isinstance(model, LoadedWD14Tagger):
        return model
    raise TypeError(
        "WD14_TAGGER is not compatible with this node. Expected a WD14 tagger "
        "loaded by Load WD14 Tagger."
    )
