# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Expose exact host-visible state while retaining one private Anima module."""

from __future__ import annotations

from copy import copy
from typing import Any

from torch import nn

_HOST_LINEAR_ATTRIBUTES = frozenset(
    {
        "bias",
        "comfy_cast_weights",
        "comfy_patched_weights",
        "in_features",
        "out_features",
        "prev_comfy_cast_weights",
        "seed_key",
        "weight",
        "_pin_state",
        "_prefetch",
        "_v",
        "_v_signature",
    }
)
_HOST_PARAMETER_NAMES = frozenset({"weight", "bias"})
_HOST_PARAMETER_SUFFIXES = (
    "_comfy_model_dtype",
    "_function",
    "_lowvram_function",
)


class AnimaHostBackedLinearModule(nn.Module):
    """Proxy installed Comfy linear host state to one exact-class shell."""

    def __getattr__(self, name: str) -> Any:
        """Resolve parameters locally and supported host state from the shell."""

        try:
            return super().__getattr__(name)
        except AttributeError as error:
            backing = self.__dict__.get("_backing")
            if (
                isinstance(backing, AnimaHostModuleBacking)
                and backing.is_linear_host_attribute(name)
                and hasattr(backing.module, name)
            ):
                return getattr(backing.module, name)
            raise error

    def __setattr__(self, name: str, value: Any) -> None:
        """Mirror direct state and installed host writes to the execution shell."""

        backing = self.__dict__.get("_backing")
        if isinstance(backing, AnimaHostModuleBacking) and (
            name in backing.direct_state_names or backing.is_linear_host_attribute(name)
        ):
            backing.replace_public_attribute(self, name, value)
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Delete supported transient host state from both public owners."""

        backing = self.__dict__.get("_backing")
        if isinstance(backing, AnimaHostModuleBacking) and (
            backing.is_linear_host_attribute(name)
        ):
            if name in backing.module.__dict__:
                delattr(backing.module, name)
            if name in self.__dict__:
                super().__delattr__(name)
            return
        super().__delattr__(name)


class AnimaHostModuleBacking:
    """Own a private execution shell and its exact public PyTorch mirror."""

    def __init__(self, module: nn.Module) -> None:
        """Clone one module shell outside the replacement's child namespace."""

        if not isinstance(module, nn.Module):
            raise TypeError("Anima host-module backing requires a PyTorch module.")
        self._module = self._execution_shell(module)
        self._direct_state_names: frozenset[str] = frozenset()

    @property
    def module(self) -> nn.Module:
        """Return the private exact-class execution shell."""

        return self._module

    @property
    def direct_state_names(self) -> frozenset[str]:
        """Return parameter and buffer names mirrored onto the replacement."""

        return self._direct_state_names

    def install_children(
        self,
        replacement: nn.Module,
        names: frozenset[str],
    ) -> None:
        """Register installed children at their unchanged public host paths."""

        for name in names:
            child = getattr(self._module, name)
            if not isinstance(child, nn.Module):
                raise TypeError(f"Anima host child {name!r} must be a PyTorch module.")
            nn.Module.__setattr__(replacement, name, child)

    def install_direct_state(self, replacement: nn.Module) -> None:
        """Mirror direct parameters and buffers without a private path prefix."""

        if (
            "weight" in self._module._parameters
            and "bias" not in self._module._parameters
        ):
            self._module._parameters["bias"] = None
        parameter_names = tuple(self._module._parameters)
        buffer_names = tuple(self._module._buffers)
        overlap = frozenset(parameter_names).intersection(buffer_names)
        if overlap:
            raise ValueError(
                "Anima host module exposes duplicate parameter/buffer names: "
                f"{sorted(overlap)!r}."
            )
        replacement._parameters = self._module._parameters
        replacement._buffers = self._module._buffers
        replacement._non_persistent_buffers_set = (
            self._module._non_persistent_buffers_set
        )
        self._module._parameters = replacement._parameters
        self._module._buffers = replacement._buffers
        self._module._non_persistent_buffers_set = (
            replacement._non_persistent_buffers_set
        )
        self._direct_state_names = frozenset((*parameter_names, *buffer_names))

    @staticmethod
    def is_linear_host_attribute(name: str) -> bool:
        """Report installed Comfy state that controls one linear's execution."""

        if name in _HOST_LINEAR_ATTRIBUTES:
            return True
        return any(
            name == f"{parameter_name}{suffix}"
            for parameter_name in _HOST_PARAMETER_NAMES
            for suffix in _HOST_PARAMETER_SUFFIXES
        )

    def replace_public_attribute(
        self,
        replacement: nn.Module,
        name: str,
        value: Any,
    ) -> None:
        """Keep the installed module and its exact public mirror synchronized."""

        setattr(self._module, name, value)
        nn.Module.__setattr__(replacement, name, value)

    @staticmethod
    def _execution_shell(module: nn.Module) -> nn.Module:
        """Copy registration maps so replacements cannot mutate the source graph."""

        shell = copy(module)
        shell._parameters = {
            name: None if parameter is None else copy(parameter)
            for name, parameter in module._parameters.items()
        }
        shell._buffers = {
            name: None if buffer is None else copy(buffer)
            for name, buffer in module._buffers.items()
        }
        shell._non_persistent_buffers_set = module._non_persistent_buffers_set.copy()
        shell._modules = module._modules.copy()
        return shell
