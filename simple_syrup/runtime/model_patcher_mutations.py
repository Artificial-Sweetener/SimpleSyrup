# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Own concrete mutations applied through ComfyUI's MODEL patcher API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, Signature, signature
from typing import cast

from comfy.patcher_extension import WrappersMP
from comfy.utils import get_attr


def _require_bound_method(
    model: object,
    method_name: str,
    positional_parameter_names: tuple[str, ...],
) -> Callable[..., object]:
    """Return a callable exposing the exact installed Comfy method signature."""

    method = getattr(model, method_name, None)
    expected = f"({', '.join(positional_parameter_names)})"
    if not callable(method):
        raise TypeError(f"MODEL must expose callable {method_name}{expected}.")
    try:
        method_signature = signature(method)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"MODEL {method_name} signature cannot be inspected; expected {expected}."
        ) from error
    if not _has_exact_positional_signature(
        method_signature,
        positional_parameter_names,
    ):
        raise TypeError(
            f"MODEL {method_name} has unsupported signature {method_signature}; "
            f"expected {expected}."
        )
    return cast(Callable[..., object], method)


def _has_exact_positional_signature(
    method_signature: Signature,
    positional_parameter_names: tuple[str, ...],
) -> bool:
    """Report whether one bound method matches the installed positional surface."""

    parameters = tuple(method_signature.parameters.values())
    return len(parameters) == len(positional_parameter_names) and all(
        parameter.name == expected_name
        and parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for parameter, expected_name in zip(
            parameters,
            positional_parameter_names,
            strict=True,
        )
    )


def _require_dictionary_attribute(
    model: object, attribute_name: str
) -> dict[object, object]:
    """Return one required mutable Comfy state dictionary without coercion."""

    value = getattr(model, attribute_name, None)
    if not isinstance(value, dict):
        raise TypeError(f"MODEL {attribute_name} must be a dictionary.")
    return value


def _require_callable_patch_list(
    patches: dict[object, object],
    patch_name: str,
) -> bool:
    """Validate an existing Comfy patch list and report whether its key exists."""

    if patch_name not in patches:
        return False
    existing = patches[patch_name]
    if not isinstance(existing, list):
        raise TypeError(f"Existing MODEL {patch_name} patches must be a list.")
    if any(not callable(patch) for patch in existing):
        raise TypeError(
            f"Existing MODEL {patch_name} patches must contain only callables."
        )
    return True


@dataclass(frozen=True)
class ModelDenoiseMaskMutation:
    """Install a denoise-mask function through the MODEL patcher API."""

    function: Callable[..., object]

    def apply(self, model: object) -> None:
        """Install the configured denoise-mask function."""

        setter = getattr(model, "set_model_denoise_mask_function", None)
        if not callable(setter):
            raise TypeError("MODEL does not support denoise-mask functions.")
        setter(self.function)


@dataclass(frozen=True)
class ModelUnetWrapperMutation:
    """Install a model-function wrapper through the MODEL patcher API."""

    wrapper: Callable[..., object]

    def apply(self, model: object) -> None:
        """Install the configured model-function wrapper."""

        setter = getattr(model, "set_model_unet_function_wrapper", None)
        if not callable(setter):
            raise TypeError("MODEL does not support model-function wrappers.")
        setter(self.wrapper)


@dataclass(frozen=True)
class ModelCalcCondBatchMutation:
    """Install a calc-cond-batch function through the MODEL patcher API."""

    function: Callable[..., object]

    def apply(self, model: object) -> None:
        """Install the configured calc-cond-batch function."""

        setter = getattr(model, "set_model_sampler_calc_cond_batch_function", None)
        if not callable(setter):
            raise TypeError("MODEL does not support calc-cond-batch functions.")
        setter(self.function)


@dataclass(frozen=True)
class ModelKeyedWrapperMutation:
    """Install one collision-safe namespaced wrapper through Comfy's keyed API."""

    wrapper_type: str
    key: str
    wrapper: Callable[..., object]

    def apply(self, model: object) -> None:
        """Validate the complete keyed-wrapper surface before installing a wrapper."""

        if not isinstance(self.wrapper_type, str) or not self.wrapper_type:
            raise ValueError("MODEL wrapper type must be a non-empty string.")
        if not isinstance(self.key, str) or not self.key.startswith("simple_syrup."):
            raise ValueError("MODEL wrapper key must start with 'simple_syrup.'.")
        if len(self.key) == len("simple_syrup."):
            raise ValueError("MODEL wrapper key must include a namespaced name.")
        if not callable(self.wrapper):
            raise TypeError("MODEL keyed wrapper must be callable.")

        getter = _require_bound_method(
            model,
            "get_wrappers",
            ("wrapper_type", "key"),
        )
        adder = _require_bound_method(
            model,
            "add_wrapper_with_key",
            ("wrapper_type", "key", "wrapper"),
        )
        existing = getter(self.wrapper_type, self.key)
        if not isinstance(existing, list):
            raise TypeError("Existing MODEL keyed wrappers must be a list.")
        if any(not callable(wrapper) for wrapper in existing):
            raise TypeError(
                "Existing MODEL keyed wrappers must contain only callables."
            )
        if existing:
            raise ValueError(f"MODEL wrapper key '{self.key}' is already installed.")
        adder(self.wrapper_type, self.key, self.wrapper)


@dataclass(frozen=True)
class ModelKeyedCallbackMutation:
    """Install one collision-safe namespaced callback through Comfy's keyed API."""

    callback_type: str
    key: str
    callback: Callable[..., object]

    def apply(self, model: object) -> None:
        """Validate the complete keyed-callback surface before installation."""

        if not isinstance(self.callback_type, str) or not self.callback_type:
            raise ValueError("MODEL callback type must be a non-empty string.")
        if not isinstance(self.key, str) or not self.key.startswith("simple_syrup."):
            raise ValueError("MODEL callback key must start with 'simple_syrup.'.")
        if len(self.key) == len("simple_syrup."):
            raise ValueError("MODEL callback key must include a namespaced name.")
        if not callable(self.callback):
            raise TypeError("MODEL keyed callback must be callable.")

        getter = _require_bound_method(
            model,
            "get_callbacks",
            ("call_type", "key"),
        )
        adder = _require_bound_method(
            model,
            "add_callback_with_key",
            ("call_type", "key", "callback"),
        )
        existing = getter(self.callback_type, self.key)
        if not isinstance(existing, list):
            raise TypeError("Existing MODEL keyed callbacks must be a list.")
        if any(not callable(callback) for callback in existing):
            raise TypeError(
                "Existing MODEL keyed callbacks must contain only callables."
            )
        if existing:
            raise ValueError(f"MODEL callback key '{self.key}' is already installed.")
        adder(self.callback_type, self.key, self.callback)


@dataclass(frozen=True)
class ModelDiffusionWrapperMutation:
    """Install one keyed wrapper at Comfy's diffusion-model wrapper boundary."""

    key: str
    wrapper: Callable[..., object]

    def apply(self, model: object) -> None:
        """Install the wrapper using the installed diffusion wrapper type."""

        ModelKeyedWrapperMutation(
            wrapper_type=WrappersMP.DIFFUSION_MODEL,
            key=self.key,
            wrapper=self.wrapper,
        ).apply(model)


@dataclass(frozen=True)
class ModelExactObjectPatchMutation:
    """Replace one exact model object after collision and identity validation."""

    path: str
    expected_object: object
    replacement: object

    def apply(self, model: object) -> None:
        """Validate exact path ownership and current identity before replacement."""

        if (
            not isinstance(self.path, str)
            or not self.path
            or any(not segment for segment in self.path.split("."))
        ):
            raise ValueError("MODEL object patch path must be a non-empty dotted path.")

        getter = _require_bound_method(model, "get_model_object", ("name",))
        adder = _require_bound_method(model, "add_object_patch", ("name", "obj"))
        object_patches = _require_dictionary_attribute(model, "object_patches")
        object_patches_backup = _require_dictionary_attribute(
            model,
            "object_patches_backup",
        )
        if self.path in object_patches:
            raise ValueError(f"MODEL object path '{self.path}' already has a patch.")
        if self.path in object_patches_backup:
            raise ValueError(
                f"MODEL object path '{self.path}' already has a patch backup."
            )

        current_object = getter(self.path)
        if current_object is not self.expected_object:
            raise ValueError(
                f"MODEL object path '{self.path}' does not match the expected object."
            )
        adder(self.path, self.replacement)


@dataclass(frozen=True)
class ModelSharedObjectPatchMutation:
    """Register one exact object patch already live across a Comfy clone handoff."""

    path: str
    expected_backup: object
    replacement: object

    def apply(self, model: object) -> None:
        """Admit only the exact shared backup and live replacement identities."""

        if (
            not isinstance(self.path, str)
            or not self.path
            or any(not segment for segment in self.path.split("."))
        ):
            raise ValueError("MODEL object patch path must be a non-empty dotted path.")
        adder = _require_bound_method(model, "add_object_patch", ("name", "obj"))
        object_patches = _require_dictionary_attribute(model, "object_patches")
        object_patches_backup = _require_dictionary_attribute(
            model,
            "object_patches_backup",
        )
        if self.path in object_patches:
            raise ValueError(f"MODEL object path '{self.path}' already has a patch.")
        if object_patches_backup.get(self.path) is not self.expected_backup:
            raise ValueError(
                f"MODEL object path '{self.path}' has a foreign shared backup."
            )
        root = getattr(model, "model", None)
        if root is None or get_attr(root, self.path) is not self.replacement:
            raise ValueError(
                f"MODEL object path '{self.path}' has a foreign live replacement."
            )
        adder(self.path, self.replacement)
