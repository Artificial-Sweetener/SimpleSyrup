# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build copy-on-write diffusion shells for persistent regional variants."""

from __future__ import annotations

from copy import copy

import torch
from torch import nn

from .standard_unet_cold_diagnostics import (
    STANDARD_UNET_COLD_PATH_DIAGNOSTICS,
    StandardUnetColdStage,
)
from .standard_unet_variant_materialization import StandardUnetMaterializedVariant


class StandardUnetVariantShellBuilder:
    """Clone only target ancestors while sharing every unchanged module."""

    def build(
        self,
        diffusion_model: object,
        variant: StandardUnetMaterializedVariant,
    ) -> nn.Module:
        """Return one executable shell without mutating the source graph."""

        if not isinstance(diffusion_model, nn.Module):
            raise TypeError("Standard UNet variant shell requires a module.")
        if not isinstance(variant, StandardUnetMaterializedVariant):
            raise TypeError(
                "Standard UNet variant shell requires materialized weights."
            )
        with STANDARD_UNET_COLD_PATH_DIAGNOSTICS.measure(
            StandardUnetColdStage.VARIANT_SHELL,
        ) as metadata:
            replacements = {
                parameter.path: parameter.tensor for parameter in variant.parameters
            }
            shell = self._clone_branch(diffusion_model, replacements, prefix="")
            metadata["region_index"] = variant.region_index
            metadata["parameter_count"] = len(variant.parameters)
        return shell

    def _clone_branch(
        self,
        module: nn.Module,
        replacements: dict[str, torch.Tensor],
        *,
        prefix: str,
    ) -> nn.Module:
        """Copy one target-bearing branch and replace its direct Parameters."""

        clone = copy(module)
        clone._modules = module._modules.copy()
        clone._parameters = module._parameters.copy()
        clone._buffers = module._buffers.copy()
        clone._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()

        direct_prefix = f"{prefix}." if prefix else ""
        for parameter_name, parameter in module._parameters.items():
            path = f"{direct_prefix}{parameter_name}"
            tensor = replacements.get(path)
            if tensor is None:
                continue
            if parameter is None or not isinstance(parameter, nn.Parameter):
                raise TypeError(f"Variant target '{path}' must be a Parameter.")
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError(f"Variant target '{path}' is incompatible.")
            clone._parameters[parameter_name] = nn.Parameter(
                tensor,
                requires_grad=False,
            )

        for child_name, child in module._modules.items():
            if child is None:
                continue
            child_prefix = f"{direct_prefix}{child_name}"
            if any(
                path == child_prefix or path.startswith(f"{child_prefix}.")
                for path in replacements
            ):
                clone._modules[child_name] = self._clone_branch(
                    child,
                    replacements,
                    prefix=child_prefix,
                )

        observed = {
            name
            for name, _parameter in clone.named_parameters()
            if name in replacements
        }
        if not prefix and observed != set(replacements):
            missing = tuple(sorted(set(replacements) - observed))
            raise ValueError(f"Variant shell did not bind target paths {missing!r}.")
        return clone


STANDARD_UNET_VARIANT_SHELL_BUILDER = StandardUnetVariantShellBuilder()
