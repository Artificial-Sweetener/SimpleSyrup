# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize ComfyUI's patcher lifecycle leak detector."""

from __future__ import annotations

import gc
import logging
import weakref
from typing import Any

import pytest
import torch

from simple_syrup.runtime.patcher_lifecycle import (
    ClipLayerMutation,
    ComfyPatcherLifecycle,
    ModelCalcCondBatchMutation,
    ModelDenoiseMaskMutation,
    ModelUnetWrapperMutation,
)


class AnimaTEModel_(torch.nn.Module):
    """Represent a tiny Anima text encoder without production weights."""

    def __init__(self) -> None:
        """Create one parameter so ComfyUI sees a normal torch module."""

        super().__init__()
        self.projection = torch.nn.Linear(1, 1)


class _AnimaClip:
    """Expose an Anima-style encoder through the native CLIP clone shape."""

    def __init__(self, patcher: Any, encoder: AnimaTEModel_) -> None:
        """Create a CLIP value around one real Comfy ModelPatcher."""

        self.patcher = patcher
        self.cond_stage_model = encoder
        self.layer_index: int | None = None

    def clone(self, disable_dynamic: bool = False) -> _AnimaClip:
        """Clone the patcher with the same behavior as Comfy's CLIP wrapper."""

        return _AnimaClip(
            self.patcher.clone(disable_dynamic=disable_dynamic),
            self.cond_stage_model,
        )

    def clip_layer(self, layer_index: int) -> None:
        """Record the selected text-encoder layer."""

        self.layer_index = layer_index


def test_real_comfy_anima_lifecycle_regression(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the unsafe predicate and every safe owner path in one worker."""

    _assert_comfy_marks_rootless_anima_patcher_dead()
    _assert_comfy_returns_clone_to_live_source()
    _assert_lifecycle_owned_anima_clip_is_safe(caplog, monkeypatch)
    _assert_supported_model_mutations_share_one_clone()


def _assert_comfy_marks_rootless_anima_patcher_dead() -> None:
    """Capture the exact condition behind ComfyUI's memory-leak warning."""

    from comfy.model_management import LoadedModel

    encoder = AnimaTEModel_()
    patcher = _patcher(encoder)
    loaded = LoadedModel(patcher)
    loaded.real_model = weakref.ref(encoder)

    del patcher
    gc.collect()

    assert loaded.model is None
    assert loaded.real_model() is encoder
    assert loaded.is_dead() is True


def _assert_comfy_returns_clone_to_live_source() -> None:
    """Prove valid clone lineage cannot satisfy the stale-patcher condition."""

    from comfy.model_management import LoadedModel

    encoder = AnimaTEModel_()
    source = _patcher(encoder)
    derived = source.clone()
    loaded = LoadedModel(derived)
    loaded.real_model = weakref.ref(encoder)

    del derived
    gc.collect()

    assert loaded.model is source
    assert loaded.is_dead() is False


def _assert_lifecycle_owned_anima_clip_is_safe(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A released first-party Anima CLIP derivation resolves to its live source."""

    import comfy.model_management
    from comfy.model_management import LoadedModel

    encoder = AnimaTEModel_()
    source_clip = _AnimaClip(_patcher(encoder), encoder)
    derived_clip = ComfyPatcherLifecycle().derive_clip(
        source_clip,
        (ClipLayerMutation(-2),),
        operation="Anima CLIP regression",
    )
    assert isinstance(derived_clip, _AnimaClip)
    assert derived_clip.layer_index == -2
    loaded = LoadedModel(derived_clip.patcher)
    loaded.real_model = weakref.ref(encoder)
    monkeypatch.setattr(comfy.model_management, "current_loaded_models", [loaded])

    del derived_clip
    gc.collect()
    with caplog.at_level(logging.INFO):
        comfy.model_management.cleanup_models_gc()

    assert loaded.model is source_clip.patcher
    assert loaded.is_dead() is False
    assert "Potential memory leak detected" not in caplog.text
    assert "WARNING, memory leak" not in caplog.text

    caplog.clear()
    released_encoder = AnimaTEModel_()
    released_source = _AnimaClip(_patcher(released_encoder), released_encoder)
    released_clip = ComfyPatcherLifecycle().derive_clip(
        released_source,
        (ClipLayerMutation(-2),),
        operation="released Anima CLIP regression",
    )
    released = LoadedModel(released_clip.patcher)
    released.real_model = weakref.ref(released_encoder)
    monkeypatch.setattr(comfy.model_management, "current_loaded_models", [released])

    del released_clip, released_source, released_encoder
    gc.collect()
    with caplog.at_level(logging.INFO):
        comfy.model_management.cleanup_models_gc()

    assert released.model is None
    assert released.real_model() is None
    assert released.is_dead() is False
    assert "Potential memory leak detected" not in caplog.text
    assert "WARNING, memory leak" not in caplog.text


def _assert_supported_model_mutations_share_one_clone() -> None:
    """All supported first-party MODEL changes share one verified derivation."""

    source = _patcher(torch.nn.Linear(1, 1))

    def denoise_mask(*args: object, **kwargs: object) -> object:
        """Return a stable test sentinel."""

        del args, kwargs
        return object()

    def model_wrapper(args: object) -> object:
        """Return the supplied model-wrapper arguments."""

        return args

    def calc_cond_batch(args: object) -> object:
        """Return the supplied calc-cond-batch arguments."""

        return args

    derived = ComfyPatcherLifecycle().derive_model(
        source,
        (
            ModelDenoiseMaskMutation(denoise_mask),
            ModelUnetWrapperMutation(model_wrapper),
            ModelCalcCondBatchMutation(calc_cond_batch),
        ),
        operation="MODEL mutation regression",
    )

    assert derived.parent is source
    assert derived.model_options["denoise_mask_function"] is denoise_mask
    assert derived.model_options["model_function_wrapper"] is model_wrapper
    assert derived.model_options["sampler_calc_cond_batch_function"] is calc_cond_batch


def test_lifecycle_rejects_a_clone_without_comfy_parent_lineage() -> None:
    """A non-Comfy clone cannot silently enter a first-party lifecycle path."""

    class BrokenModel:
        """Return an unrelated object from clone()."""

        def clone(self) -> BrokenModel:
            """Return a clone without a parent link."""

            return BrokenModel()

    with pytest.raises(RuntimeError, match="without its source as parent"):
        ComfyPatcherLifecycle().derive_model(
            BrokenModel(),
            (),
            operation="broken regression",
        )


def test_lifecycle_preserves_vae_identity() -> None:
    """The first-party VAE contract never clones or mutates the supplied value."""

    vae = object()

    result = ComfyPatcherLifecycle().preserve_vae(
        vae,
        operation="VAE regression",
    )

    assert result is vae


def _patcher(model: torch.nn.Module) -> Any:
    """Create a CPU patcher without loading model weights."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
