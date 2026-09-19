# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify automatic NegPiP dispatch on real Comfy patcher objects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from comfy.model_base import SDXL, Anima, BaseModel, Krea2, SDXLRefiner
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
from comfy.sd import CLIP

from simple_syrup.runtime.negpip.anima import (
    WRAPPER_KEY as ANIMA_WRAPPER_KEY,
)
from simple_syrup.runtime.negpip.anima import (
    anima_attn2_negpip,
)
from simple_syrup.runtime.negpip.krea2 import (
    CLIP_MARKER,
    KREA_TOKEN_KEY,
    Krea2NegpipTokenizer,
    krea2_attn1_negpip,
)
from simple_syrup.runtime.negpip.krea2 import (
    WRAPPER_KEY as KREA_WRAPPER_KEY,
)
from simple_syrup.runtime.negpip.standard import standard_attn2_negpip
from simple_syrup.services.negpip_model_service import (
    MODEL_MARKER,
    NegpipModelService,
)


class _Encoder(torch.nn.Module):
    """Expose the encoder method patched by the standard NegPiP path."""

    def encode_token_weights(self, pairs: object) -> object:
        """Return the supplied placeholder pairs."""

        return pairs


class _ClipRoot(torch.nn.Module):
    """Provide the model object structure used by supported CLIP families."""

    def __init__(self, *, krea: bool = False) -> None:
        """Install either the standard or Krea encoder surface."""

        super().__init__()
        if krea:
            setattr(self, KREA_TOKEN_KEY, _Encoder())
        else:
            self.clip_l = _Encoder()

    def encode_token_weights(
        self,
        pairs: object,
        *,
        template_end: int = -1,
    ) -> tuple[object, None, dict[str, object]]:
        """Stand in for Krea's root shape-preserving encoder."""

        del template_end
        return pairs, None, {}


class _Tokenizer:
    """Represent the installed tokenizer retained by a Krea proxy."""


@pytest.mark.parametrize("model_class", (BaseModel, SDXL, SDXLRefiner))
def test_service_patches_every_standard_ppm_family(
    model_class: type[BaseModel],
) -> None:
    """SD1, SDXL, and SDXL Refiner receive one cloned PPM-equivalent path."""

    model = _model_patcher(model_class)
    clip = _clip(krea=False)

    prepared_model, prepared_clip = NegpipModelService().prepare(model, clip)

    assert isinstance(prepared_model, ModelPatcher)
    assert isinstance(prepared_clip, CLIP)
    assert prepared_model is not model
    assert prepared_clip is not clip
    assert MODEL_MARKER not in model.model_options
    assert MODEL_MARKER not in clip.patcher.model_options
    assert prepared_model.model_options[MODEL_MARKER] is True
    assert prepared_clip.patcher.model_options[MODEL_MARKER] is True
    assert _attention_patch(prepared_model, "attn2_patch") is standard_attn2_negpip
    assert "clip_l.encode_token_weights" in prepared_clip.patcher.object_patches


def test_service_patches_anima_with_mask_wrapper_and_attention() -> None:
    """Anima receives its extra condition, diffusion wrapper, and V patch."""

    prepared_model, prepared_clip = NegpipModelService().prepare(
        _model_patcher(Anima),
        _clip(krea=False),
    )

    model = cast(ModelPatcher, prepared_model)
    clip = cast(CLIP, prepared_clip)
    assert model.model_options[MODEL_MARKER] is True
    assert clip.patcher.model_options[MODEL_MARKER] is True
    assert "extra_conds" in model.object_patches
    assert _attention_patch(model, "attn2_patch") is anima_attn2_negpip
    assert ANIMA_WRAPPER_KEY in model.wrappers[WrappersMP.DIFFUSION_MODEL]


def test_service_patches_krea_without_claiming_ppm_clip_encoding() -> None:
    """Krea uses its layered encoder proxy and joint attn1 value patch."""

    source_clip = _clip(krea=True)
    prepared_model, prepared_clip = NegpipModelService().prepare(
        _model_patcher(Krea2),
        source_clip,
    )

    model = cast(ModelPatcher, prepared_model)
    clip = cast(CLIP, prepared_clip)
    assert model.model_options[MODEL_MARKER] is True
    assert MODEL_MARKER not in clip.patcher.model_options
    assert clip.patcher.model_options[CLIP_MARKER] is True
    assert isinstance(clip.tokenizer, Krea2NegpipTokenizer)
    assert clip.tokenizer is not source_clip.tokenizer
    assert "encode_token_weights" in clip.patcher.object_patches
    assert "extra_conds" in model.object_patches
    assert _attention_patch(model, "attn1_patch") is krea2_attn1_negpip
    assert KREA_WRAPPER_KEY in model.wrappers[WrappersMP.DIFFUSION_MODEL]


def test_service_reuses_already_patched_pair_without_double_patching() -> None:
    """An existing PPM model marker makes automatic preparation idempotent."""

    model = _model_patcher(BaseModel)
    clip = _clip(krea=False)
    model.model_options[MODEL_MARKER] = True

    prepared_model, prepared_clip = NegpipModelService().prepare(model, clip)

    assert prepared_model is model
    assert prepared_clip is clip
    prepared_model_typed = cast(ModelPatcher, prepared_model)
    transformer_options = cast(
        dict[str, object],
        prepared_model_typed.model_options["transformer_options"],
    )
    assert "patches" not in transformer_options


def _model_patcher(model_class: type[BaseModel]) -> ModelPatcher:
    """Construct an unloaded family instance behind Comfy's real patcher."""

    model = object.__new__(model_class)
    torch.nn.Module.__init__(model)
    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)


def _clip(*, krea: bool) -> CLIP:
    """Construct a cloneable unloaded CLIP around a real model patcher."""

    clip = CLIP(no_init=True)
    root = _ClipRoot(krea=krea)
    device = torch.device("cpu")
    clip.patcher = ModelPatcher(root, load_device=device, offload_device=device)
    clip.cond_stage_model = root
    clip.tokenizer = _Tokenizer()
    clip.layer_idx = None
    clip.tokenizer_options = {}
    clip.use_clip_schedule = False
    clip.apply_hooks_to_conds = None
    return clip


def _attention_patch(model: ModelPatcher, key: str) -> Callable[..., Any]:
    """Return the single installed attention patch from model options."""

    transformer_options = cast(
        dict[str, object], model.model_options["transformer_options"]
    )
    patches = cast(dict[str, list[Callable[..., Any]]], transformer_options["patches"])
    assert len(patches[key]) == 1
    return patches[key][0]
