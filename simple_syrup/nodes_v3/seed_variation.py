# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose seed variation as a Comfy v3 MODEL patch node."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.seed_variation import (
    MAX_SEED,
    MAX_VARIATION_STRENGTH,
    MIN_SEED,
    MIN_VARIATION_STRENGTH,
)
from ..nodes import tooltips
from ..services.seed_variation_model_service import SEED_VARIATION_MODEL_SERVICE

if TYPE_CHECKING:

    class _ComfyNodeBase:
        """Type-checking base for Comfy v3 nodes."""

        RETURN_TYPES: ClassVar[list[str]]
        RETURN_NAMES: ClassVar[list[str]]

else:
    _ComfyNodeBase = import_module("comfy_api.latest").io.ComfyNode

_comfy_io: Any = None if TYPE_CHECKING else import_module("comfy_api.latest").io


class SeedVariationV3(_ComfyNodeBase):
    """Derive a MODEL that varies sampler-provided initial noise."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the seed-variation MODEL patch contract."""

        return _comfy_io.Schema(
            node_id="SimpleSyrup.SeedVariation",
            display_name="Seed Variation",
            category="SimpleSyrup/Sampling",
            description=(
                "Creates related generations by mixing sampler noise toward a "
                "second deterministic seed."
            ),
            search_aliases=["variation seed", "subseed", "seed interpolation"],
            inputs=[
                _comfy_io.Model.Input(
                    "model",
                    tooltip=tooltips.SEED_VARIATION_MODEL_INPUT,
                ),
                _comfy_io.Int.Input(
                    "variation_seed",
                    default=0,
                    min=MIN_SEED,
                    max=MAX_SEED,
                    control_after_generate=True,
                    tooltip=tooltips.VARIATION_SEED,
                ),
                _comfy_io.Float.Input(
                    "variation_strength",
                    default=0.0,
                    min=MIN_VARIATION_STRENGTH,
                    max=MAX_VARIATION_STRENGTH,
                    step=0.01,
                    round=0.01,
                    tooltip=tooltips.VARIATION_STRENGTH,
                ),
            ],
            outputs=[
                _comfy_io.Model.Output(
                    "model",
                    tooltip=tooltips.SEED_VARIATION_MODEL_OUTPUT,
                )
            ],
        )

    @classmethod
    def execute(
        cls,
        model: object,
        variation_seed: int,
        variation_strength: float,
    ) -> tuple[object]:
        """Return the source or a MODEL carrying initial-noise variation."""

        return (
            SEED_VARIATION_MODEL_SERVICE.prepare(
                model=model,
                variation_seed=variation_seed,
                variation_strength=variation_strength,
            ),
        )
