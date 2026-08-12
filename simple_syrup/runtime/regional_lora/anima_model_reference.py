# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Retain non-owning Anima model identity and immutable patch geometry."""

from __future__ import annotations

from weakref import ReferenceType, ref

from torch import nn


class AnimaDiffusionModelReference:
    """Validate wrapper ownership without extending the model's lifetime."""

    def __init__(self, model: nn.Module) -> None:
        """Capture weak identity without extending the model's lifetime."""

        if not isinstance(model, nn.Module):
            raise TypeError("Anima model reference requires a PyTorch module.")
        self._model: ReferenceType[nn.Module] = ref(model)

    @property
    def patch_temporal(self) -> int:
        """Return the current validated temporal patch size."""

        return self._positive_integer("patch_temporal")

    @property
    def patch_spatial(self) -> int:
        """Return the current validated spatial patch size."""

        return self._positive_integer("patch_spatial")

    def owns(self, candidate: object) -> bool:
        """Report whether the live referenced model is the exact candidate."""

        return self._model() is candidate

    def _positive_integer(self, attribute: str) -> int:
        """Read one positive patch dimension from the live referenced model."""

        model = self._model()
        if model is None:
            raise RuntimeError("Referenced Anima diffusion model was released.")
        value = getattr(model, attribute, None)
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"Installed Anima {attribute} must be a positive integer; "
                f"observed {value!r}."
            )
        return value
