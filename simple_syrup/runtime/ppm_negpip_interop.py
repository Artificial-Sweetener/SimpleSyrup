# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adapt the complete installed PPM NegPiP patch family to regional execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import torch
from comfy.patcher_extension import WrappersMP

from ..domain.regional_model_capabilities import RegionalModelFamily

_MODEL_MARKER = "ppm_negpip"
_ANIMA_WRAPPER_KEY = "ppm_negpip_anima"
_ANIMA_CONDITION_KEY = "c_ppm_negpip_mask"
_ANIMA_TRANSFORMER_KEY = "ppm_negpip_mask"
_EXTRA_CONDS_PATH = "extra_conds"
_ATTN2_PATCH_NAME = "attn2_patch"
_UNET_CALLBACK = (
    "src.negpip.unet_negpip",
    "sdxl_attn2_negpip",
)
_ANIMA_CALLBACK = (
    "src.negpip.anima_negpip",
    "cosmos_attn2_negpip",
)
_ANIMA_WRAPPER = (
    "src.negpip.anima_negpip",
    "cosmos_diffusion_negpip_wrapper",
)
_ANIMA_EXTRA_CONDS = (
    "src.negpip.anima_negpip",
    "anima_extra_conds_negpip_wrapper.<locals>._anima_extra_conds_negpip_wrapper",
)


class PpmNegpipSemantics(StrEnum):
    """Identify the family-specific NegPiP conditioning representation."""

    STANDARD_UNET_SPLIT_KEY_VALUE = "standard-unet-split-key-value"
    ANIMA_VALUE_MASK = "anima-value-mask"


@dataclass(frozen=True, slots=True)
class PpmNegpipInterop:
    """Retain identity-validated PPM objects needed by regional execution."""

    semantics: PpmNegpipSemantics
    attention_patch: Callable[..., object]

    def __post_init__(self) -> None:
        """Require one typed semantic mode and callable preserved callback."""

        if not isinstance(self.semantics, PpmNegpipSemantics):
            raise TypeError("NegPiP semantics have an invalid type.")
        if not callable(self.attention_patch):
            raise TypeError("NegPiP attention patch must be callable.")

    def extract_value_multiplier(
        self,
        model_conditions: dict[object, object],
        context: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return one validated Anima value multiplier or no UNet multiplier."""

        if self.semantics is PpmNegpipSemantics.STANDARD_UNET_SPLIT_KEY_VALUE:
            return None
        condition = model_conditions.get(_ANIMA_CONDITION_KEY)
        if condition is None:
            return context.new_ones((*context.shape[:2], 1))
        multiplier = getattr(condition, "cond", None)
        if not isinstance(multiplier, torch.Tensor):
            raise TypeError("Anima NegPiP value mask condition must contain a tensor.")
        if (
            multiplier.ndim != 3
            or int(multiplier.shape[0]) != int(context.shape[0])
            or int(multiplier.shape[1]) != int(context.shape[1])
            or int(multiplier.shape[2]) != 1
        ):
            raise ValueError(
                "Anima NegPiP value mask must match the conditioning batch and "
                "sequence with one multiplier channel."
            )
        multiplier = multiplier.to(device=context.device, dtype=context.dtype)
        if not bool(((multiplier == 1) | (multiplier == -1)).all().item()):
            raise ValueError("Anima NegPiP value mask must contain only -1 and 1.")
        return multiplier

    def prepare_anima_transformer_options(
        self,
        source: dict[str, object],
        packed_multiplier: torch.Tensor,
    ) -> dict[str, object]:
        """Publish a packed mask on an isolated Anima cross-attention call."""

        if self.semantics is not PpmNegpipSemantics.ANIMA_VALUE_MASK:
            raise ValueError("Only Anima NegPiP semantics can publish a value mask.")
        if not isinstance(packed_multiplier, torch.Tensor):
            raise TypeError("Packed Anima NegPiP multiplier must be a tensor.")
        prepared = source.copy()
        prepared[_ANIMA_TRANSFORMER_KEY] = packed_multiplier
        return prepared


class PpmNegpipInteropValidator:
    """Admit only one complete identity-validated PPM NegPiP family."""

    def validate(
        self,
        family: RegionalModelFamily,
        *,
        model_options: dict[object, object],
        wrappers: dict[str, dict[object, list[object]]],
        object_patches: dict[object, object],
        transformer_patches: dict[str, list[object]],
    ) -> PpmNegpipInterop | None:
        """Return preserved NegPiP state or reject every partial/conflicting form."""

        if not isinstance(family, RegionalModelFamily):
            raise TypeError("NegPiP interop requires a model family.")
        marker = model_options.get(_MODEL_MARKER, False)
        if not isinstance(marker, bool):
            raise TypeError("MODEL ppm_negpip marker must be boolean.")
        attention = transformer_patches.get(_ATTN2_PATCH_NAME, [])
        anima_wrappers = wrappers.get(WrappersMP.DIFFUSION_MODEL, {}).get(
            _ANIMA_WRAPPER_KEY,
            [],
        )
        extra_conds = object_patches.get(_EXTRA_CONDS_PATH)
        recognized_surface = any(
            (
                any(_is_identity(item, *_UNET_CALLBACK) for item in attention),
                any(_is_identity(item, *_ANIMA_CALLBACK) for item in attention),
                bool(anima_wrappers),
                _is_identity(extra_conds, *_ANIMA_EXTRA_CONDS),
            )
        )
        if not marker:
            if recognized_surface:
                raise ValueError(
                    "MODEL contains an incomplete NegPiP patch family without its "
                    "marker. Reapply CLIP NegPip to a clean MODEL."
                )
            return None
        if family is RegionalModelFamily.STANDARD_UNET:
            return self._validate_standard_unet(
                attention,
                anima_wrappers=anima_wrappers,
                extra_conds=extra_conds,
            )
        if family is RegionalModelFamily.ANIMA:
            return self._validate_anima(
                attention,
                anima_wrappers=anima_wrappers,
                extra_conds=extra_conds,
            )
        raise ValueError(f"NegPiP does not support model family {family.value!r}.")

    @staticmethod
    def _validate_standard_unet(
        attention: list[object],
        *,
        anima_wrappers: list[object],
        extra_conds: object,
    ) -> PpmNegpipInterop:
        """Require exactly PPM's single UNet split-K/V callback surface."""

        if (
            len(attention) != 1
            or not _is_identity(attention[0], *_UNET_CALLBACK)
            or anima_wrappers
            or _is_identity(extra_conds, *_ANIMA_EXTRA_CONDS)
        ):
            raise ValueError(
                "Standard UNet NegPiP requires exactly its PPM split-K/V attention "
                "patch and no Anima NegPiP surfaces."
            )
        return PpmNegpipInterop(
            PpmNegpipSemantics.STANDARD_UNET_SPLIT_KEY_VALUE,
            cast(Callable[..., object], attention[0]),
        )

    @staticmethod
    def _validate_anima(
        attention: list[object],
        *,
        anima_wrappers: list[object],
        extra_conds: object,
    ) -> PpmNegpipInterop:
        """Require PPM's exact callback, wrapper, and extra-condition surfaces."""

        if (
            len(attention) != 1
            or not _is_identity(attention[0], *_ANIMA_CALLBACK)
            or len(anima_wrappers) != 1
            or not _is_identity(anima_wrappers[0], *_ANIMA_WRAPPER)
            or not _is_identity(extra_conds, *_ANIMA_EXTRA_CONDS)
        ):
            raise ValueError(
                "Anima NegPiP requires exactly its PPM attention patch, keyed "
                "diffusion wrapper, and extra_conds object patch."
            )
        return PpmNegpipInterop(
            PpmNegpipSemantics.ANIMA_VALUE_MASK,
            cast(Callable[..., object], attention[0]),
        )


def _is_identity(
    value: object,
    module_suffix: str,
    qualified_name: str,
) -> bool:
    """Match one callable by its stable defining module suffix and qualified name."""

    if not callable(value):
        return False
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    return (
        isinstance(module, str)
        and (module == module_suffix or module.endswith(f".{module_suffix}"))
        and qualname == qualified_name
    )


PPM_NEGPIP_INTEROP_VALIDATOR = PpmNegpipInteropValidator()
