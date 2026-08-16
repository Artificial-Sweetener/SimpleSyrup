# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize ComfyUI's patcher lifecycle leak detector."""

from __future__ import annotations

import gc
import logging
import weakref
from typing import Any, cast

import pytest
import torch

from simple_syrup.runtime.patcher_lifecycle import ComfyPatcherLifecycle
from simple_syrup.runtime.regional_lora.execution_cache import ModelCloneLineage


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


def test_clip_alignment_precedes_mutations_after_dynamic_to_static_clone() -> None:
    """Mutate the same independently reloaded encoder the returned CLIP executes."""

    source_encoder = AnimaTEModel_()
    reloaded_encoder = AnimaTEModel_()

    class ReloadingPatcher:
        """Represent Comfy's dynamic patcher creating a static model delegate."""

        def __init__(self) -> None:
            """Expose the source model without a parent."""

            self.model = source_encoder
            self.parent: object | None = None

        def clone(self, disable_dynamic: bool = False) -> ReloadingPatcher:
            """Return an independent model only for the static clone request."""

            assert disable_dynamic is True
            derived = ReloadingPatcher()
            derived.model = reloaded_encoder
            derived.parent = self
            return derived

    observed: list[tuple[object, object]] = []

    class ObserveAlignedClip:
        """Record the encoder identities visible at mutation time."""

        def apply(self, clip: object) -> None:
            """Capture the returned CLIP and patcher model identities."""

            clip_boundary = cast(Any, clip)
            observed.append(
                (
                    clip_boundary.cond_stage_model,
                    clip_boundary.patcher.model,
                )
            )

    source_clip = _AnimaClip(ReloadingPatcher(), source_encoder)
    derived = ComfyPatcherLifecycle().derive_clip(
        source_clip,
        (ObserveAlignedClip(),),
        operation="dynamic-to-static CLIP regression",
        disable_dynamic=True,
    )

    assert derived.cond_stage_model is reloaded_encoder
    assert derived.patcher.model is reloaded_encoder
    assert observed == [(reloaded_encoder, reloaded_encoder)]


def test_model_lifecycle_requests_static_clone_explicitly() -> None:
    """Preserve direct lineage across Comfy's dynamic-to-static MODEL boundary."""

    class CloneableModel:
        """Record the requested Comfy clone mode."""

        def __init__(self) -> None:
            """Create one root MODEL value."""

            self.parent: object | None = None
            self.disable_dynamic: bool | None = None

        def clone(self, disable_dynamic: bool = False) -> CloneableModel:
            """Return one direct child and record the mode on the source."""

            self.disable_dynamic = disable_dynamic
            derived = CloneableModel()
            derived.parent = self
            return derived

    source = CloneableModel()
    derived = ComfyPatcherLifecycle().derive_model(
        source,
        (),
        operation="static MODEL characterization",
        disable_dynamic=True,
    )

    assert source.disable_dynamic is True
    assert derived.parent is source


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
        (),
        operation="Anima CLIP regression",
    )
    assert isinstance(derived_clip, _AnimaClip)
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
        (),
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
    """Ordered MODEL mutations receive the same verified derivation."""

    source = _patcher(torch.nn.Linear(1, 1))
    applications: list[tuple[str, object]] = []

    class RecordingMutation:
        """Record ordered lifecycle mutation application."""

        def __init__(self, name: str) -> None:
            """Store the mutation name."""

            self.name = name

        def apply(self, model: object) -> None:
            """Record the derived model and mutation order."""

            applications.append((self.name, model))

    derived = ComfyPatcherLifecycle().derive_model(
        source,
        (
            RecordingMutation("first"),
            RecordingMutation("second"),
            RecordingMutation("third"),
        ),
        operation="MODEL mutation regression",
    )

    assert derived.parent is source
    assert [name for name, _model in applications] == ["first", "second", "third"]
    assert all(model is derived for _name, model in applications)


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


def test_lifecycle_clones_hooks_without_returning_the_source() -> None:
    """Keep HookGroup-like cloning inside the single lifecycle authority."""

    class CloneableHooks:
        """Return one distinct typed hook value."""

        def clone(self) -> CloneableHooks:
            """Return a distinct clone."""

            return CloneableHooks()

    source = CloneableHooks()

    result = ComfyPatcherLifecycle().clone_hooks(
        source,
        operation="hook regression",
    )

    assert isinstance(result, CloneableHooks)
    assert result is not source


def test_regional_lora_lineage_does_not_retain_model_patcher() -> None:
    """Cache identity must not keep a released Comfy model patcher alive."""

    source = _patcher(torch.nn.Linear(1, 1))
    derived = source.clone()
    derived_reference = weakref.ref(derived)

    lineage = ModelCloneLineage.from_model(derived)
    expected = (derived.clone_base_uuid, derived.patches_uuid)

    del derived
    gc.collect()

    assert derived_reference() is None
    assert (lineage.clone_base_uuid, lineage.patches_uuid) == expected


def _patcher(model: torch.nn.Module) -> Any:
    """Create a CPU patcher without loading model weights."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
