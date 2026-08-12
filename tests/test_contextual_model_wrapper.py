"""Verify Contextual Diffusion wrapper delegation and fusion behavior."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from simple_syrup.domain.contextual_diffusion import (
    ContextualDiffusionControls,
    build_contextual_diffusion_plan,
)
from simple_syrup.domain.spatial_views import SpatialBatchLayout, SpatialViewKind
from simple_syrup.runtime import contextual_model_wrapper as wrapper_module
from simple_syrup.runtime.contextual_model_wrapper import (
    ContextualDiffusionModelWrapper,
)
from simple_syrup.runtime.sampling_model_types import ApplyModel
from simple_syrup.runtime.spatial_model_arguments import make_spatial_view_model_args


def test_preserved_attention_wrapper_receives_every_local_tile_layout() -> None:
    """Route all Contextual local work through existing spatial model layouts."""

    layouts: list[SpatialBatchLayout] = []

    def attention_wrapper(
        apply_model: ApplyModel,
        args: dict[str, Any],
    ) -> torch.Tensor:
        """Record the canonical layout consumed by a prepared attention model."""

        transformer_options = args["c"]["transformer_options"]
        layout = transformer_options["simple_syrup"]["spatial_batch_layout"]
        assert isinstance(layout, SpatialBatchLayout)
        layouts.append(layout)
        return apply_model(args["input"], args["timestep"], **args["c"])

    wrapper = _wrapper(global_weight=0.0, existing_wrapper=attention_wrapper)
    output = wrapper(
        lambda x, _timestep, **_conditioning: torch.ones_like(x),
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {"transformer_options": {}},
        },
    )

    assert torch.equal(output, torch.ones_like(output))
    assert len(layouts) == 1
    layout = layouts[0]
    assert layout.input_batch_size == 1
    assert layout.view_count == 2
    assert all(view.kind is SpatialViewKind.TILE for view in layout.views)
    assert tuple((view.source_x, view.source_y) for view in layout.views) == (
        (0, 0),
        (16, 0),
    )


def test_preserved_attention_wrapper_receives_the_reduced_global_layout() -> None:
    """Route local and reduced-global work through the same prepared model chain."""

    layouts: list[SpatialBatchLayout] = []

    def attention_wrapper(
        apply_model: ApplyModel,
        args: dict[str, Any],
    ) -> torch.Tensor:
        """Record each canonical layout reaching the shared attention model."""

        transformer_options = args["c"]["transformer_options"]
        layout = transformer_options["simple_syrup"]["spatial_batch_layout"]
        assert isinstance(layout, SpatialBatchLayout)
        layouts.append(layout)
        return apply_model(args["input"], args["timestep"], **args["c"])

    output = _wrapper(existing_wrapper=attention_wrapper)(
        lambda x, _timestep, **_conditioning: torch.ones_like(x),
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {"transformer_options": {}},
        },
    )

    assert torch.equal(output, torch.ones_like(output))
    assert len(layouts) == 2
    assert all(view.kind is SpatialViewKind.TILE for view in layouts[0].views)
    global_layout = layouts[1]
    assert global_layout.input_batch_size == layouts[0].input_batch_size == 1
    assert global_layout.view_count == 1
    assert global_layout.views[0].kind is SpatialViewKind.CONTEXTUAL_GLOBAL
    assert (
        global_layout.views[0].source_width,
        global_layout.views[0].source_height,
        global_layout.views[0].model_width,
        global_layout.views[0].model_height,
    ) == (32, 16, 16, 8)


def _controls(
    *, global_weight: float = 1.0, global_steps: int = 1
) -> ContextualDiffusionControls:
    """Return a deterministic two-tile wrapper configuration."""

    return ContextualDiffusionControls(16, 0, 2, global_weight, global_steps, 0.5)


def _wrapper(
    *,
    global_weight: float = 1.0,
    global_steps: int = 1,
    diffusion_mode: str = "multidiffusion",
    existing_wrapper: Any = None,
) -> ContextualDiffusionModelWrapper:
    """Build one wrapper over a 32-by-16 canvas."""

    controls = _controls(global_weight=global_weight, global_steps=global_steps)
    plan = build_contextual_diffusion_plan(
        latent_width=32,
        latent_height=16,
        controls=controls,
        segs=None,
    )
    return ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0]),
        diffusion_mode=diffusion_mode,
        existing_wrapper=existing_wrapper,
    )


def test_existing_wrapper_owns_every_underlying_model_evaluation() -> None:
    """Chain local and global calls through the preserved wrapper in order."""

    shapes: list[tuple[int, ...]] = []

    def existing(apply_model: object, args: dict[str, Any]) -> torch.Tensor:
        """Capture transformed calls without invoking the raw model."""

        del apply_model
        tensor = args["input"]
        assert isinstance(tensor, torch.Tensor)
        shapes.append(tuple(tensor.shape))
        return torch.ones_like(tensor)

    def forbidden(*args: object, **kwargs: object) -> torch.Tensor:
        """Fail if wrapper chaining is bypassed."""

        del args, kwargs
        raise AssertionError("raw model bypassed existing wrapper")

    output = _wrapper(existing_wrapper=existing)(
        forbidden,
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {},
        },
    )

    assert shapes == [(2, 1, 16, 16), (1, 1, 8, 16)]
    assert torch.equal(output, torch.ones_like(output))


def test_global_call_uses_one_full_source_reduced_model_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct the canonical global layout from the active model batch."""

    layouts: list[SpatialBatchLayout] = []
    transform = make_spatial_view_model_args

    def capture_layout(
        *,
        args: dict[str, Any],
        layout: SpatialBatchLayout,
    ) -> dict[str, Any]:
        """Capture and apply the global model-argument layout."""

        layouts.append(layout)
        return transform(args=args, layout=layout)

    monkeypatch.setattr(
        wrapper_module,
        "make_spatial_view_model_args",
        capture_layout,
    )
    _wrapper()(
        lambda x, _timestep, **_conditioning: torch.zeros_like(x),
        {
            "input": torch.zeros((2, 1, 16, 32)),
            "timestep": torch.ones((2,)),
            "c": {},
        },
    )

    assert len(layouts) == 1
    layout = layouts[0]
    assert (layout.canvas_width, layout.canvas_height) == (32, 16)
    assert layout.input_batch_size == 2
    assert layout.view_count == 1
    assert layout.views[0].kind is SpatialViewKind.CONTEXTUAL_GLOBAL
    assert (
        layout.views[0].source_width,
        layout.views[0].source_height,
        layout.views[0].model_width,
        layout.views[0].model_height,
    ) == (32, 16, 16, 8)


def test_off_plan_shape_delegates_without_spatial_transformation() -> None:
    """Leave unrelated model calls untouched for wrapper composition."""

    received: list[torch.Tensor] = []

    def apply_model(
        x: torch.Tensor, timestep: torch.Tensor, **conditioning: object
    ) -> torch.Tensor:
        """Record the unmodified direct call."""

        del timestep, conditioning
        received.append(x)
        return x + 2

    x = torch.zeros((1, 1, 8, 8))
    output = _wrapper()(
        apply_model, {"input": x, "timestep": torch.tensor([1.0]), "c": {}}
    )

    assert received == [x]
    assert torch.equal(output, x + 2)


def test_wrapper_rejects_non_tensor_model_input() -> None:
    """Fail before layout or model evaluation for malformed input."""

    with pytest.raises(ValueError, match="model input must be a tensor"):
        _wrapper()(lambda *_args, **_kwargs: torch.tensor(0), {"input": "latent"})


def test_direct_delegation_rejects_non_mapping_conditioning() -> None:
    """Validate the raw apply-model boundary when no prior wrapper exists."""

    controls = _controls()
    plan = build_contextual_diffusion_plan(
        latent_width=16, latent_height=16, controls=controls, segs=None
    )
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0]),
        existing_wrapper=None,
    )

    with pytest.raises(ValueError, match="conditioning must be a dict"):
        wrapper(
            lambda *_args, **_kwargs: torch.tensor(0),
            {
                "input": torch.zeros((1, 1, 16, 16)),
                "timestep": torch.tensor([1.0]),
                "c": [],
            },
        )


@pytest.mark.parametrize(("global_weight", "global_steps"), [(0.0, 1), (1.0, 0)])
def test_disabled_global_authority_performs_only_local_evaluation(
    global_weight: float, global_steps: int
) -> None:
    """Skip the reduced-global call when either authority control disables it."""

    calls: list[tuple[int, int]] = []

    def apply_model(
        x: torch.Tensor, timestep: torch.Tensor, **conditioning: object
    ) -> torch.Tensor:
        """Record each evaluated spatial shape."""

        del timestep, conditioning
        calls.append((int(x.shape[-2]), int(x.shape[-1])))
        return torch.ones_like(x)

    _wrapper(global_weight=global_weight, global_steps=global_steps)(
        apply_model,
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {},
        },
    )

    assert calls == [(16, 16)]


def test_bcdhw_global_correction_preserves_singleton_depth() -> None:
    """Apply the same correction across final spatial axes of Anima-style latents."""

    def apply_model(
        x: torch.Tensor, timestep: torch.Tensor, **conditioning: object
    ) -> torch.Tensor:
        """Return distinct local and reduced-global constants."""

        del timestep, conditioning
        value = 3.0 if x.shape[-2:] == (8, 16) else 1.0
        return torch.full_like(x, value)

    output = _wrapper()(
        apply_model,
        {
            "input": torch.zeros((2, 4, 1, 16, 32)),
            "timestep": torch.ones((2,)),
            "c": {},
        },
    )

    assert output.shape == (2, 4, 1, 16, 32)
    assert torch.allclose(output, torch.full_like(output, 3.0))


def test_global_prediction_replaces_low_frequency_without_removing_local_detail() -> (
    None
):
    """Preserve the exact local plus weighted global-minus-local-low formula."""

    def apply_model(
        x: torch.Tensor, timestep: torch.Tensor, **conditioning: object
    ) -> torch.Tensor:
        """Return global intent or alternating local detail by view shape."""

        del timestep, conditioning
        if x.shape[-2:] == (8, 16):
            return torch.full_like(x, 3.0)
        rows = torch.arange(x.shape[-2], device=x.device).reshape(1, 1, -1, 1)
        return torch.where(rows % 2 == 0, 1.0, -1.0).expand_as(x)

    output = _wrapper()(
        apply_model,
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {},
        },
    )

    assert torch.allclose(output[:, :, 0::2], torch.full((1, 1, 8, 32), 4.0))
    assert torch.allclose(output[:, :, 1::2], torch.full((1, 1, 8, 32), 2.0))


@pytest.mark.parametrize("diffusion_mode", ["multidiffusion", "mixture_of_diffusers"])
def test_weighted_correction_formula_is_exact_for_both_local_fusion_modes(
    diffusion_mode: str,
) -> None:
    """Apply local plus weight times global-canvas minus local-low exactly."""

    def apply_model(
        x: torch.Tensor,
        timestep: torch.Tensor,
        **conditioning: object,
    ) -> torch.Tensor:
        """Return global authority or zero-mean alternating local detail."""

        del timestep, conditioning
        if x.shape[-2:] == (8, 16):
            return torch.full_like(x, 3.0)
        rows = torch.arange(x.shape[-2], device=x.device).reshape(1, 1, -1, 1)
        return torch.where(rows % 2 == 0, 1.0, -1.0).expand_as(x)

    output = _wrapper(
        global_weight=0.5,
        diffusion_mode=diffusion_mode,
    )(
        apply_model,
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {},
        },
    )

    assert torch.allclose(output[:, :, 0::2], torch.full((1, 1, 8, 32), 2.5))
    assert torch.allclose(output[:, :, 1::2], torch.full((1, 1, 8, 32), 0.5))


def test_local_and_global_calls_receive_complete_reference_latents() -> None:
    """Keep independent reference images intact through both spatial views."""

    reference = torch.arange(1 * 4 * 16 * 32, dtype=torch.float32).reshape(
        (1, 4, 16, 32)
    )
    received: list[torch.Tensor] = []

    def apply_model(
        x: torch.Tensor, timestep: torch.Tensor, **conditioning: object
    ) -> torch.Tensor:
        """Capture the reference batch presented to each prediction."""

        del timestep
        references = conditioning["ref_latents"]
        assert isinstance(references, list)
        assert isinstance(references[0], torch.Tensor)
        received.append(references[0])
        return torch.zeros_like(x)

    _wrapper()(
        apply_model,
        {
            "input": torch.zeros((1, 1, 16, 32)),
            "timestep": torch.tensor([1.0]),
            "c": {"ref_latents": [reference]},
        },
    )

    assert len(received) == 2
    assert torch.equal(received[0], torch.cat((reference, reference), dim=0))
    assert torch.equal(received[1], reference)


def test_one_tile_plan_delegates_to_one_original_evaluation() -> None:
    """Avoid contextual transformation when the canvas already fits one tile."""

    controls = _controls()
    plan = build_contextual_diffusion_plan(
        latent_width=16, latent_height=16, controls=controls, segs=None
    )
    wrapper = ContextualDiffusionModelWrapper(
        plan=plan,
        controls=controls,
        sigmas=torch.tensor([1.0, 0.0]),
        existing_wrapper=None,
    )
    calls = 0

    def apply_model(
        x: torch.Tensor, timestep: torch.Tensor, **conditioning: object
    ) -> torch.Tensor:
        """Count direct model evaluations."""

        del timestep, conditioning
        nonlocal calls
        calls += 1
        return x + 2

    x = torch.zeros((1, 1, 16, 16))
    output = wrapper(
        apply_model,
        {"input": x, "timestep": torch.tensor([1.0]), "c": {}},
    )

    assert calls == 1
    assert torch.equal(output, x + 2)
