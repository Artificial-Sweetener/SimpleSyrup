# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove isolated per-call Anima activation geometry publication."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import comfy.ops
import pytest
import torch
from comfy.ldm.anima.model import Anima
from comfy.patcher_extension import WrapperExecutor
from torch import nn

from simple_syrup.domain.spatial_views import (
    SpatialBatchLayout,
    SpatialView,
    SpatialViewKind,
)
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE
from simple_syrup.runtime.regional_lora.anima_activation_context import (
    ANIMA_ACTIVATION_GEOMETRY_KEY,
    ANIMA_ACTIVATION_WRAPPER_KEY,
    AnimaActivationContext,
    AnimaActivationDiffusionWrapper,
    AnimaActivationGeometry,
    anima_activation_wrapper_mutation,
)
from simple_syrup.runtime.regional_lora.anima_module_surface import (
    ANIMA_MODULE_SURFACE_DISCOVERY,
    AnimaModuleSurface,
)
from simple_syrup.runtime.regional_lora.anima_targets import ANIMA_BLOCK_COUNT
from simple_syrup.runtime.spatial_model_arguments import (
    SIMPLE_SYRUP_TRANSFORMER_NAMESPACE,
    SPATIAL_BATCH_LAYOUT_KEY,
)


class _RecordingExecutor:
    """Expose Comfy's class-aware callable executor surface for focused tests."""

    def __init__(
        self,
        class_obj: object,
        callback: Callable[[tuple[object, ...], dict[str, object]], object],
    ) -> None:
        """Retain the installed model identity and downstream callback."""

        self.class_obj = class_obj
        self._callback = callback

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Forward the exact wrapper arguments to the test callback."""

        return self._callback(args, kwargs)


@pytest.fixture(scope="module")
def anima_surface() -> AnimaModuleSurface:
    """Discover one real installed Anima graph backed by meta-device weights."""

    model = Anima(
        max_img_h=2,
        max_img_w=2,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,
        num_blocks=ANIMA_BLOCK_COUNT,
        num_heads=16,
        mlp_ratio=4.0,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        extra_per_block_abs_pos_emb=False,
        device=torch.device("meta"),
        dtype=torch.float16,
        operations=comfy.ops.disable_weight_init,
    )
    return ANIMA_MODULE_SURFACE_DISCOVERY.discover(model)


def test_wrapper_publishes_full_context_geometry_without_changing_output(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Publish exact B/T/H/W and patch grid through copied task-local state."""

    context = AnimaActivationContext()
    wrapper = AnimaActivationDiffusionWrapper(anima_surface, context)
    model_input = torch.zeros((2, 16, 1, 8, 10))
    output = torch.ones((1,))
    source_options: dict[str, object] = {
        "foreign": object(),
        SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {"preserved": object()},
    }
    observed_geometry: AnimaActivationGeometry | None = None
    observed_options: dict[object, object] | None = None

    def downstream(args: tuple[object, ...], kwargs: dict[str, object]) -> torch.Tensor:
        """Capture published metadata while the task-local context is active."""

        nonlocal observed_geometry, observed_options
        assert args[0] is model_input
        options = kwargs["transformer_options"]
        assert isinstance(options, dict)
        observed_options = options
        observed_geometry = context.require_current()
        assert context.from_transformer_options(options) is observed_geometry
        return output

    executor = _RecordingExecutor(anima_surface.diffusion_model, downstream)

    result = wrapper(
        executor,
        model_input,
        torch.ones((2,)),
        torch.zeros((2, 512, 1024)),
        None,
        None,
        transformer_options=source_options,
    )

    assert result is output
    assert observed_geometry == AnimaActivationGeometry(
        input_batch_size=2,
        activation_time=1,
        activation_height=8,
        activation_width=10,
        patch_temporal=1,
        patch_spatial=2,
        query_time=1,
        query_height=4,
        query_width=5,
        spatial_layout=None,
    )
    assert observed_geometry.query_token_count == 20
    assert observed_options is not None
    assert observed_options is not source_options
    assert observed_options["foreign"] is source_options["foreign"]
    source_namespace = source_options[SIMPLE_SYRUP_TRANSFORMER_NAMESPACE]
    assert isinstance(source_namespace, dict)
    assert ANIMA_ACTIVATION_GEOMETRY_KEY not in source_namespace
    assert context.current_or_none() is None


def test_wrapper_retains_and_validates_current_spatial_layout_identity(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Retain the authoritative view layout across repeated conditioning chunks."""

    context = AnimaActivationContext()
    wrapper = AnimaActivationDiffusionWrapper(anima_surface, context)
    layout = _two_tile_layout(input_batch_size=2)
    model_input = torch.zeros((8, 16, 1, 8, 8))
    options = {SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {SPATIAL_BATCH_LAYOUT_KEY: layout}}

    def downstream(
        args: tuple[object, ...], kwargs: dict[str, object]
    ) -> AnimaActivationGeometry:
        """Return the active geometry from the downstream diffusion boundary."""

        del args, kwargs
        return context.require_current()

    geometry = wrapper(
        _RecordingExecutor(anima_surface.diffusion_model, downstream),
        model_input,
        transformer_options=options,
    )

    assert isinstance(geometry, AnimaActivationGeometry)
    assert geometry.spatial_layout is layout
    assert geometry.input_batch_size == 8
    assert options[SIMPLE_SYRUP_TRANSFORMER_NAMESPACE] == {
        SPATIAL_BATCH_LAYOUT_KEY: layout
    }


def test_wrapper_restores_nested_activation_context(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Restore outer geometry after a nested diffusion evaluation completes."""

    context = AnimaActivationContext()
    wrapper = AnimaActivationDiffusionWrapper(anima_surface, context)
    outer_input = torch.zeros((1, 16, 1, 8, 8))
    inner_input = torch.zeros((1, 16, 1, 12, 10))
    observations: list[tuple[int, int]] = []

    def inner(args: tuple[object, ...], kwargs: dict[str, object]) -> torch.Tensor:
        """Record the nested call's distinct geometry."""

        del args, kwargs
        geometry = context.require_current()
        observations.append((geometry.activation_height, geometry.activation_width))
        return torch.ones((1,))

    def outer(args: tuple[object, ...], kwargs: dict[str, object]) -> torch.Tensor:
        """Record outer geometry before and after the nested wrapper call."""

        del args, kwargs
        before = context.require_current()
        observations.append((before.activation_height, before.activation_width))
        result = wrapper(
            _RecordingExecutor(anima_surface.diffusion_model, inner),
            inner_input,
        )
        after = context.require_current()
        observations.append((after.activation_height, after.activation_width))
        assert isinstance(result, torch.Tensor)
        return result

    wrapper(
        _RecordingExecutor(anima_surface.diffusion_model, outer),
        outer_input,
    )

    assert observations == [(8, 8), (12, 10), (8, 8)]
    assert context.current_or_none() is None


def test_wrapper_isolates_concurrent_activation_contexts(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Keep simultaneous model-call geometry isolated between worker contexts."""

    context = AnimaActivationContext()
    wrapper = AnimaActivationDiffusionWrapper(anima_surface, context)
    barrier = Barrier(2)

    def evaluate(height: int, width: int) -> tuple[int, int]:
        """Hold one active wrapper scope while another worker enters its own."""

        def downstream(
            args: tuple[object, ...], kwargs: dict[str, object]
        ) -> tuple[int, int]:
            """Return geometry after both worker contexts are active."""

            del args, kwargs
            barrier.wait(timeout=5)
            geometry = context.require_current()
            return geometry.activation_height, geometry.activation_width

        result = wrapper(
            _RecordingExecutor(anima_surface.diffusion_model, downstream),
            torch.zeros((1, 16, 1, height, width)),
        )
        if not isinstance(result, tuple):
            raise AssertionError("Concurrent wrapper must return a geometry tuple")
        return result

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(evaluate, 8, 10)
        second = workers.submit(evaluate, 12, 14)

    assert {first.result(), second.result()} == {(8, 10), (12, 14)}
    assert context.current_or_none() is None


def test_clone_local_mutation_preserves_existing_wrapper_order_and_classes(
    anima_surface: AnimaModuleSurface,
) -> None:
    """Append one namespaced wrapper on a clone without global class mutation."""

    context = AnimaActivationContext()
    source = _patcher(nn.Linear(1, 1))
    events: list[str] = []

    def upstream(
        executor: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        """Record a preserved wrapper around the namespaced activation wrapper."""

        events.append("upstream-enter")
        result = executor(*args, **kwargs)
        events.append("upstream-exit")
        return result

    source.add_wrapper_with_key("diffusion_model", "upstream.owner", upstream)
    original_anima_forward = Anima.forward
    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        (anima_activation_wrapper_mutation(anima_surface, context),),
        operation="Anima activation geometry publication",
    )
    installed = derived.get_wrappers("diffusion_model", ANIMA_ACTIVATION_WRAPPER_KEY)

    def inner(*args: object, **kwargs: object) -> str:
        """Observe activation state at the innermost diffusion call."""

        del args, kwargs
        assert context.require_current().query_token_count == 16
        events.append("inner")
        return "prediction"

    result = WrapperExecutor.new_class_executor(
        inner,
        anima_surface.diffusion_model,
        derived.get_all_wrappers("diffusion_model"),
    ).execute(torch.zeros((1, 16, 1, 8, 8)))

    assert result == "prediction"
    assert source.get_wrappers("diffusion_model", ANIMA_ACTIVATION_WRAPPER_KEY) == []
    assert len(installed) == 1
    assert isinstance(installed[0], AnimaActivationDiffusionWrapper)
    assert derived.get_all_wrappers("diffusion_model")[0] is upstream
    assert events == ["upstream-enter", "inner", "upstream-exit"]
    assert Anima.forward is original_anima_forward


def _two_tile_layout(*, input_batch_size: int) -> SpatialBatchLayout:
    """Build two view-major 8x8 tiles over one 16x8 latent canvas."""

    return SpatialBatchLayout(
        canvas_width=16,
        canvas_height=8,
        views=(
            SpatialView(
                SpatialViewKind.TILE,
                source_x=0,
                source_y=0,
                source_width=8,
                source_height=8,
                model_width=8,
                model_height=8,
            ),
            SpatialView(
                SpatialViewKind.TILE,
                source_x=8,
                source_y=0,
                source_width=8,
                source_height=8,
                model_width=8,
                model_height=8,
            ),
        ),
        input_batch_size=input_batch_size,
    )


@pytest.mark.parametrize(
    ("model_input", "options", "message"),
    [
        (torch.zeros((1, 16, 8, 8)), {}, "BxCxTxHxW"),
        (torch.zeros((1, 16, 2, 8, 8)), {}, "requires image.*T=1"),
        (
            torch.zeros((1, 16, 1, 8, 8)),
            {SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: object()},
            "simple_syrup must be a dictionary",
        ),
        (
            torch.zeros((1, 16, 1, 8, 8)),
            {SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {SPATIAL_BATCH_LAYOUT_KEY: object()}},
            "must be a SpatialBatchLayout",
        ),
        (
            torch.zeros((3, 16, 1, 8, 8)),
            {
                SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {
                    SPATIAL_BATCH_LAYOUT_KEY: _two_tile_layout(input_batch_size=2)
                }
            },
            "whole number of conditioning chunks",
        ),
        (
            torch.zeros((4, 16, 1, 10, 8)),
            {
                SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {
                    SPATIAL_BATCH_LAYOUT_KEY: _two_tile_layout(input_batch_size=2)
                }
            },
            "H/W must match every spatial view",
        ),
        (
            torch.zeros((1, 16, 1, 8, 8)),
            {
                SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {
                    ANIMA_ACTIVATION_GEOMETRY_KEY: object()
                }
            },
            "already present",
        ),
    ],
)
def test_wrapper_rejects_malformed_activation_and_namespace_inputs(
    anima_surface: AnimaModuleSurface,
    model_input: torch.Tensor,
    options: dict[str, object],
    message: str,
) -> None:
    """Fail before downstream model work for every invalid geometry boundary."""

    called = False

    def downstream(args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        """Record a downstream call that invalid input must prevent."""

        nonlocal called
        del args, kwargs
        called = True

    wrapper = AnimaActivationDiffusionWrapper(
        anima_surface,
        AnimaActivationContext(),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        wrapper(
            _RecordingExecutor(anima_surface.diffusion_model, downstream),
            model_input,
            transformer_options=options,
        )

    assert called is False


def test_wrapper_rejects_wrong_model_identity_and_patch_sizes(
    anima_surface: AnimaModuleSurface,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind execution to the discovered object and positive installed patch sizes."""

    wrapper = AnimaActivationDiffusionWrapper(
        anima_surface,
        AnimaActivationContext(),
    )
    executor = _RecordingExecutor(object(), lambda args, kwargs: None)
    with pytest.raises(ValueError, match="does not own the discovered"):
        wrapper(executor, torch.zeros((1, 16, 1, 8, 8)))

    monkeypatch.setattr(anima_surface.diffusion_model, "patch_spatial", 0)
    valid_executor = _RecordingExecutor(
        anima_surface.diffusion_model,
        lambda args, kwargs: None,
    )
    with pytest.raises(ValueError, match="patch_spatial must be a positive integer"):
        wrapper(valid_executor, torch.zeros((1, 16, 1, 8, 8)))


def test_context_reader_rejects_missing_or_malformed_publication() -> None:
    """Make downstream consumers fail closed on absent namespaced geometry."""

    context = AnimaActivationContext()

    with pytest.raises(RuntimeError, match="unavailable outside"):
        context.require_current()
    with pytest.raises(TypeError, match="must be a dictionary"):
        context.from_transformer_options(None)
    with pytest.raises(TypeError, match="simple_syrup must be a dictionary"):
        context.from_transformer_options({})
    with pytest.raises(TypeError, match="require published"):
        context.from_transformer_options({SIMPLE_SYRUP_TRANSFORMER_NAMESPACE: {}})


def _patcher(model: nn.Module) -> Any:
    """Create a real CPU Comfy MODEL patcher for clone-isolation tests."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
