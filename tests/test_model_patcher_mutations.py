"""Characterize concrete MODEL patcher mutation behavior."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import pytest
import torch

from simple_syrup.runtime.model_patcher_mutations import (
    ModelAttn2PatchesMutation,
    ModelCalcCondBatchMutation,
    ModelDenoiseMaskMutation,
    ModelDiffusionWrapperMutation,
    ModelExactObjectPatchMutation,
    ModelKeyedWrapperMutation,
    ModelUnetWrapperMutation,
)
from simple_syrup.runtime.patcher_lifecycle import PATCHER_LIFECYCLE


class _ModelMutation(Protocol):
    """Describe the concrete mutation boundary under test."""

    def apply(self, model: object) -> None:
        """Apply one mutation to a MODEL value."""


class _RecordingModel:
    """Record calls through each supported Comfy MODEL setter."""

    def __init__(self) -> None:
        """Initialize empty mutation state."""

        self.denoise_mask: Callable[..., object] | None = None
        self.unet_wrapper: Callable[..., object] | None = None
        self.calc_cond_batch: Callable[..., object] | None = None

    def set_model_denoise_mask_function(
        self,
        function: Callable[..., object],
    ) -> None:
        """Record the denoise-mask function."""

        self.denoise_mask = function

    def set_model_unet_function_wrapper(
        self,
        wrapper: Callable[..., object],
    ) -> None:
        """Record the model-function wrapper."""

        self.unet_wrapper = wrapper

    def set_model_sampler_calc_cond_batch_function(
        self,
        function: Callable[..., object],
    ) -> None:
        """Record the calc-cond-batch function."""

        self.calc_cond_batch = function


def test_model_mutations_use_exact_comfy_setter_surface() -> None:
    """Install every supported MODEL mutation without altering its callable."""

    model = _RecordingModel()

    def denoise_mask(*args: object, **kwargs: object) -> object:
        """Return a stable denoise sentinel."""

        del args, kwargs
        return object()

    def unet_wrapper(args: object) -> object:
        """Return supplied wrapper arguments."""

        return args

    def calc_cond_batch(args: object) -> object:
        """Return supplied conditioning arguments."""

        return args

    ModelDenoiseMaskMutation(denoise_mask).apply(model)
    ModelUnetWrapperMutation(unet_wrapper).apply(model)
    ModelCalcCondBatchMutation(calc_cond_batch).apply(model)

    assert model.denoise_mask is denoise_mask
    assert model.unet_wrapper is unet_wrapper
    assert model.calc_cond_batch is calc_cond_batch


def test_model_mutations_integrate_with_real_comfy_patcher() -> None:
    """Install every supported mutation in a real Comfy MODEL options mapping."""

    model = _patcher(torch.nn.Linear(1, 1))

    def denoise_mask(*args: object, **kwargs: object) -> object:
        """Return a stable denoise sentinel."""

        del args, kwargs
        return object()

    def unet_wrapper(args: object) -> object:
        """Return supplied wrapper arguments."""

        return args

    def calc_cond_batch(args: object) -> object:
        """Return supplied conditioning arguments."""

        return args

    ModelDenoiseMaskMutation(denoise_mask).apply(model)
    ModelUnetWrapperMutation(unet_wrapper).apply(model)
    ModelCalcCondBatchMutation(calc_cond_batch).apply(model)

    assert model.model_options["denoise_mask_function"] is denoise_mask
    assert model.model_options["model_function_wrapper"] is unet_wrapper
    assert model.model_options["sampler_calc_cond_batch_function"] is calc_cond_batch


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            ModelDenoiseMaskMutation(lambda: None),
            "MODEL does not support denoise-mask functions",
        ),
        (
            ModelUnetWrapperMutation(lambda: None),
            "MODEL does not support model-function wrappers",
        ),
        (
            ModelCalcCondBatchMutation(lambda: None),
            "MODEL does not support calc-cond-batch functions",
        ),
    ],
)
def test_model_mutations_reject_missing_comfy_setter(
    mutation: _ModelMutation,
    message: str,
) -> None:
    """Fail closed when a MODEL lacks the required public mutation surface."""

    with pytest.raises(TypeError, match=message):
        mutation.apply(object())


def test_collision_safe_mutations_integrate_through_one_real_comfy_clone() -> None:
    """Preserve source state and external wrappers while installing every surface."""

    from comfy.model_patcher import ModelPatcher

    source = _patcher(torch.nn.Linear(1, 1))

    def external_wrapper(*args: object) -> tuple[object, ...]:
        """Return external wrapper arguments."""

        return args

    def keyed_wrapper(*args: object) -> tuple[object, ...]:
        """Return keyed wrapper arguments."""

        return args

    def diffusion_wrapper(*args: object) -> tuple[object, ...]:
        """Return diffusion wrapper arguments."""

        return args

    def input_patch(*args: object) -> tuple[object, ...]:
        """Return attn2 input arguments."""

        return args

    def output_patch(*args: object) -> tuple[object, ...]:
        """Return attn2 output arguments."""

        return args

    expected_weight = source.get_model_object("weight")
    replacement_weight = torch.nn.Parameter(torch.ones_like(expected_weight))
    source.add_wrapper_with_key("external", "other.extension", external_wrapper)
    original_add_wrapper = ModelPatcher.add_wrapper_with_key
    original_attn2_setter = ModelPatcher.set_model_attn2_patch

    derived = PATCHER_LIFECYCLE.derive_model(
        source,
        (
            ModelKeyedWrapperMutation(
                "custom",
                "simple_syrup.custom",
                keyed_wrapper,
            ),
            ModelDiffusionWrapperMutation(
                "simple_syrup.diffusion",
                diffusion_wrapper,
            ),
            ModelAttn2PatchesMutation(input_patch, output_patch),
            ModelExactObjectPatchMutation(
                "weight",
                expected_weight,
                replacement_weight,
            ),
        ),
        operation="collision-safe mutation regression",
    )

    assert derived.parent is source
    assert source.get_wrappers("external", "other.extension") == [external_wrapper]
    assert source.get_wrappers("custom", "simple_syrup.custom") == []
    assert source.get_wrappers("diffusion_model", "simple_syrup.diffusion") == []
    assert source.model_options["transformer_options"].get("patches") is None
    assert source.object_patches == {}
    assert derived.get_wrappers("external", "other.extension") == [external_wrapper]
    assert derived.get_wrappers("custom", "simple_syrup.custom") == [keyed_wrapper]
    assert derived.get_wrappers("diffusion_model", "simple_syrup.diffusion") == [
        diffusion_wrapper
    ]
    derived_patches = derived.model_options["transformer_options"]["patches"]
    assert derived_patches["attn2_patch"] == [input_patch]
    assert derived_patches["attn2_output_patch"] == [output_patch]
    assert derived.object_patches["weight"] is replacement_weight
    assert ModelPatcher.add_wrapper_with_key is original_add_wrapper
    assert ModelPatcher.set_model_attn2_patch is original_attn2_setter


@pytest.mark.parametrize(
    ("wrapper_type", "key", "wrapper", "message"),
    [
        ("", "simple_syrup.valid", lambda: None, "non-empty string"),
        ("custom", "foreign.key", lambda: None, "must start"),
        ("custom", "simple_syrup.", lambda: None, "namespaced name"),
        ("custom", "simple_syrup.valid", None, "must be callable"),
    ],
)
def test_keyed_wrapper_mutation_validates_every_owned_field(
    wrapper_type: str,
    key: str,
    wrapper: Any,
    message: str,
) -> None:
    """Reject malformed wrapper values without modifying the real patcher."""

    model = _patcher(torch.nn.Linear(1, 1))

    with pytest.raises((TypeError, ValueError), match=message):
        ModelKeyedWrapperMutation(wrapper_type, key, wrapper).apply(model)

    assert model.wrappers == {}


def test_keyed_wrapper_mutation_rejects_malformed_prior_wrapper_before_duplicate() -> (
    None
):
    """Diagnose corrupt existing wrapper state before reporting a key collision."""

    model = _patcher(torch.nn.Linear(1, 1))
    malformed_wrapper = object()
    model.wrappers = {"custom": {"simple_syrup.key": [malformed_wrapper]}}

    with pytest.raises(TypeError, match="contain only callables"):
        ModelKeyedWrapperMutation(
            "custom",
            "simple_syrup.key",
            lambda: None,
        ).apply(model)

    assert model.wrappers["custom"]["simple_syrup.key"] == [malformed_wrapper]


def test_keyed_wrapper_mutation_rejects_duplicate_without_appending() -> None:
    """Treat any valid existing namespaced wrapper as an ownership collision."""

    model = _patcher(torch.nn.Linear(1, 1))

    def existing() -> None:
        """Provide one pre-existing wrapper."""

    model.add_wrapper_with_key("custom", "simple_syrup.key", existing)

    with pytest.raises(ValueError, match="already installed"):
        ModelKeyedWrapperMutation(
            "custom",
            "simple_syrup.key",
            lambda: None,
        ).apply(model)

    assert model.get_wrappers("custom", "simple_syrup.key") == [existing]


def test_keyed_wrapper_mutation_rejects_changed_signature_before_adding() -> None:
    """Fail closed when Comfy's keyed API no longer matches the installed baseline."""

    class ChangedSurface:
        """Expose a deliberately changed getter signature."""

        def __init__(self) -> None:
            """Initialize an untouched call log."""

            self.added = False

        def get_wrappers(self, kind: str, key: str) -> list[Callable[..., object]]:
            """Return no wrappers through an unsupported parameter name."""

            del kind, key
            return []

        def add_wrapper_with_key(
            self,
            wrapper_type: str,
            key: str,
            wrapper: Callable[..., object],
        ) -> None:
            """Record an add call that must never occur."""

            del wrapper_type, key, wrapper
            self.added = True

    model = ChangedSurface()

    with pytest.raises(TypeError, match="unsupported signature"):
        ModelKeyedWrapperMutation(
            "custom",
            "simple_syrup.key",
            lambda: None,
        ).apply(model)

    assert model.added is False


def test_keyed_wrapper_mutation_rejects_non_list_getter_result() -> None:
    """Reject an incompatible keyed-wrapper state container before adding."""

    class ChangedState:
        """Expose exact methods but an unsupported wrapper collection."""

        def __init__(self) -> None:
            """Initialize an untouched call log."""

            self.added = False

        def get_wrappers(self, wrapper_type: str, key: str) -> tuple[object, ...]:
            """Return an unsupported immutable collection."""

            del wrapper_type, key
            return ()

        def add_wrapper_with_key(
            self,
            wrapper_type: str,
            key: str,
            wrapper: Callable[..., object],
        ) -> None:
            """Record an add call that must never occur."""

            del wrapper_type, key, wrapper
            self.added = True

    model = ChangedState()

    with pytest.raises(TypeError, match="must be a list"):
        ModelKeyedWrapperMutation(
            "custom",
            "simple_syrup.key",
            lambda: None,
        ).apply(model)

    assert model.added is False


def test_attn2_mutation_rejects_output_collision_before_installing_input() -> None:
    """Validate both attn2 slots before making the paired mutation non-atomic."""

    model = _patcher(torch.nn.Linear(1, 1))

    def existing(*args: object) -> tuple[object, ...]:
        """Return pre-existing attn2 output arguments."""

        return args

    model.set_model_attn2_output_patch(existing)

    with pytest.raises(ValueError, match="output patch is already installed"):
        ModelAttn2PatchesMutation(lambda *args: args, lambda *args: args).apply(model)

    patches = model.model_options["transformer_options"]["patches"]
    assert "attn2_patch" not in patches
    assert patches["attn2_output_patch"] == [existing]


@pytest.mark.parametrize(
    ("patch_name", "existing", "message"),
    [
        ("attn2_patch", object(), "must be a list"),
        ("attn2_output_patch", [object()], "contain only callables"),
    ],
)
def test_attn2_mutation_rejects_malformed_prior_state(
    patch_name: str,
    existing: object,
    message: str,
) -> None:
    """Reject every malformed existing attn2 slot before either setter runs."""

    model = _patcher(torch.nn.Linear(1, 1))
    model.model_options["transformer_options"]["patches"] = {patch_name: existing}

    with pytest.raises(TypeError, match=message):
        ModelAttn2PatchesMutation(lambda *args: args, lambda *args: args).apply(model)

    assert model.model_options["transformer_options"]["patches"] == {
        patch_name: existing
    }
    model.model_options["transformer_options"]["patches"] = {}


@pytest.mark.parametrize(
    ("input_patch", "output_patch", "message"),
    [
        (None, lambda: None, "input patch must be callable"),
        (lambda: None, None, "output patch must be callable"),
    ],
)
def test_attn2_mutation_validates_both_callbacks(
    input_patch: Any,
    output_patch: Any,
    message: str,
) -> None:
    """Reject non-callable attn2 callbacks without creating patch state."""

    model = _patcher(torch.nn.Linear(1, 1))

    with pytest.raises(TypeError, match=message):
        ModelAttn2PatchesMutation(input_patch, output_patch).apply(model)

    assert model.model_options["transformer_options"].get("patches") is None


@pytest.mark.parametrize(
    ("model_options", "message"),
    [
        (None, "model_options must be a dictionary"),
        ({"transformer_options": None}, "transformer_options must be a dictionary"),
        ({"transformer_options": {"patches": None}}, "patches must be a dictionary"),
    ],
)
def test_attn2_mutation_rejects_malformed_state_containers(
    model_options: object,
    message: str,
) -> None:
    """Reject malformed Comfy option containers without invoking either setter."""

    class MalformedState:
        """Expose exact setters around deliberately malformed state."""

        def __init__(self, state: object) -> None:
            """Store malformed state and an empty setter call log."""

            self.model_options = state
            self.calls: list[str] = []

        def set_model_attn2_patch(self, patch: Callable[..., object]) -> None:
            """Record an input call that must never occur."""

            del patch
            self.calls.append("input")

        def set_model_attn2_output_patch(
            self,
            patch: Callable[..., object],
        ) -> None:
            """Record an output call that must never occur."""

            del patch
            self.calls.append("output")

    model = MalformedState(model_options)

    with pytest.raises(TypeError, match=message):
        ModelAttn2PatchesMutation(lambda: None, lambda: None).apply(model)

    assert model.calls == []


def test_attn2_mutation_rejects_changed_output_signature_before_input_call() -> None:
    """Validate both installed setter signatures before installing either patch."""

    class ChangedSurface:
        """Expose one exact and one changed attn2 setter."""

        def __init__(self) -> None:
            """Initialize valid state and an empty setter call log."""

            self.model_options: dict[str, object] = {"transformer_options": {}}
            self.calls: list[str] = []

        def set_model_attn2_patch(self, patch: Callable[..., object]) -> None:
            """Record an input call that must never occur."""

            del patch
            self.calls.append("input")

        def set_model_attn2_output_patch(
            self,
            callback: Callable[..., object],
        ) -> None:
            """Expose the deliberately changed parameter name."""

            del callback
            self.calls.append("output")

    model = ChangedSurface()

    with pytest.raises(TypeError, match="unsupported signature"):
        ModelAttn2PatchesMutation(lambda: None, lambda: None).apply(model)

    assert model.calls == []


@pytest.mark.parametrize("path", ["", ".weight", "weight.", "model..weight"])
def test_exact_object_patch_rejects_invalid_dotted_paths(path: str) -> None:
    """Reject empty path segments before inspecting or changing patcher state."""

    model = _patcher(torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match="non-empty dotted path"):
        ModelExactObjectPatchMutation(path, object(), object()).apply(model)

    assert model.object_patches == {}


@pytest.mark.parametrize("state_name", ["object_patches", "object_patches_backup"])
def test_exact_object_patch_rejects_existing_active_or_backup_state(
    state_name: str,
) -> None:
    """Reject both forms of an already-owned exact object path."""

    model = _patcher(torch.nn.Linear(1, 1))
    expected = model.get_model_object("weight")
    getattr(model, state_name)["weight"] = expected

    with pytest.raises(ValueError, match="already has a patch"):
        ModelExactObjectPatchMutation("weight", expected, object()).apply(model)

    assert (
        "weight" not in model.object_patches
        or model.object_patches["weight"] is expected
    )


def test_exact_object_patch_rejects_changed_current_identity() -> None:
    """Require the caller's exact expected object before claiming the path."""

    model = _patcher(torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match="does not match the expected object"):
        ModelExactObjectPatchMutation("weight", object(), object()).apply(model)

    assert model.object_patches == {}


def test_exact_object_patch_rejects_changed_adder_signature_before_lookup() -> None:
    """Validate the whole object-patch API before looking up or replacing an object."""

    class ChangedSurface:
        """Expose an incompatible object-patch adder."""

        def __init__(self) -> None:
            """Initialize valid state and untouched lookup state."""

            self.object_patches: dict[str, object] = {}
            self.object_patches_backup: dict[str, object] = {}
            self.looked_up = False

        def get_model_object(self, name: str) -> object:
            """Record a lookup that must never occur."""

            del name
            self.looked_up = True
            return object()

        def add_object_patch(self, path: str, value: object) -> None:
            """Expose deliberately changed parameter names."""

            del path, value

    model = ChangedSurface()

    with pytest.raises(TypeError, match="unsupported signature"):
        ModelExactObjectPatchMutation("weight", object(), object()).apply(model)

    assert model.looked_up is False
    assert model.object_patches == {}


@pytest.mark.parametrize("attribute_name", ["object_patches", "object_patches_backup"])
def test_exact_object_patch_rejects_malformed_state_dictionary(
    attribute_name: str,
) -> None:
    """Require both exact-path collision stores to remain dictionaries."""

    model = _patcher(torch.nn.Linear(1, 1))
    expected = model.get_model_object("weight")
    setattr(model, attribute_name, None)

    with pytest.raises(TypeError, match=f"{attribute_name} must be a dictionary"):
        ModelExactObjectPatchMutation("weight", expected, object()).apply(model)

    setattr(model, attribute_name, {})


def _patcher(model: torch.nn.Module) -> Any:
    """Create a real CPU Comfy MODEL patcher."""

    from comfy.model_patcher import ModelPatcher

    device = torch.device("cpu")
    return ModelPatcher(model, load_device=device, offload_device=device)
