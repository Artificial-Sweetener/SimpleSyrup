"""Represent the external MODEL patcher surface for real spatial wrapper tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from simple_syrup.runtime.sampling_model_types import ModelFunctionWrapper


class SpatialModel:
    """Keep independent model options and an explicit lifecycle parent."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        """Initialize the minimal host state needed by wrapper derivation."""
        self.model_options = {} if options is None else options
        self.load_device = torch.device("cpu")
        self.parent: SpatialModel | None = None

    def clone(self) -> SpatialModel:
        """Copy host options and retain the direct source identity."""
        derived = SpatialModel(self.model_options.copy())
        derived.parent = self
        return derived

    def set_model_unet_function_wrapper(self, wrapper: ModelFunctionWrapper) -> None:
        """Implement the public Comfy wrapper installation boundary."""
        self.model_options["model_function_wrapper"] = wrapper

    def set_model_sampler_calc_cond_batch_function(
        self, function: Callable[..., object]
    ) -> None:
        """Implement the external Comfy regional prediction registration boundary."""
        self.model_options["sampler_calc_cond_batch_function"] = function
