# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify loaded-call native forward override and exact restoration."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from simple_syrup.runtime.regional_lora.operation_call_scope import (
    RegionalCallScopedOperation,
    RegionalOperationCallScope,
)


class _DiffusionGraph(nn.Module):
    """Expose one statically typed nested operation path."""

    linear: nn.Module

    def __init__(self, linear: nn.Module) -> None:
        """Register the supplied native operation."""

        super().__init__()
        self.linear = linear


class _RootGraph(nn.Module):
    """Expose one statically typed diffusion-model owner."""

    diffusion_model: _DiffusionGraph

    def __init__(self, diffusion_model: _DiffusionGraph) -> None:
        """Register the supplied diffusion graph."""

        super().__init__()
        self.diffusion_model = diffusion_model


def test_scope_overrides_only_forward_and_preserves_registered_graph() -> None:
    """Keep native object and child registration unchanged during execution."""

    root, original, replacement = _graph()
    scope = RegionalOperationCallScope(
        root,
        (RegionalCallScopedOperation("diffusion_model.linear", original, replacement),),
    )

    inputs = torch.tensor([[1.0, 2.0]])
    native_output = original(inputs)
    registered = dict(root.diffusion_model.named_children())

    with scope.activate():
        assert root.diffusion_model.linear is original
        assert dict(root.diffusion_model.named_children()) == registered
        torch.testing.assert_close(root.diffusion_model.linear(inputs), inputs)

    assert root.diffusion_model.linear is original
    assert dict(root.diffusion_model.named_children()) == registered
    torch.testing.assert_close(root.diffusion_model.linear(inputs), native_output)


def test_scope_restores_exact_identity_after_body_failure() -> None:
    """Restore every native operation when denoising raises."""

    root, original, replacement = _graph()
    scope = RegionalOperationCallScope(
        root,
        (RegionalCallScopedOperation("diffusion_model.linear", original, replacement),),
    )

    with pytest.raises(RuntimeError, match="body failure"):
        with scope.activate():
            assert root.diffusion_model.linear is original
            raise RuntimeError("body failure")

    assert root.diffusion_model.linear is original


def test_scope_restores_preexisting_instance_forward_exactly() -> None:
    """Preserve a native instance-owned forward instead of deleting it."""

    root, original, replacement = _graph()

    def native_forward(inputs: torch.Tensor) -> torch.Tensor:
        """Return a recognizable preexisting native result."""

        return inputs + 3.0

    object.__setattr__(original, "forward", native_forward)
    scope = RegionalOperationCallScope(
        root,
        (RegionalCallScopedOperation("diffusion_model.linear", original, replacement),),
    )
    inputs = torch.tensor([[1.0, 2.0]])

    with scope.activate():
        torch.testing.assert_close(original(inputs), inputs)

    assert original.__dict__["forward"] is native_forward
    torch.testing.assert_close(original(inputs), inputs + 3.0)


def test_scope_fails_before_mutation_when_native_identity_changed() -> None:
    """Reject a stale graph without partially installing replacements."""

    root, original, replacement = _graph()
    scope = RegionalOperationCallScope(
        root,
        (RegionalCallScopedOperation("diffusion_model.linear", original, replacement),),
    )
    foreign = nn.Linear(2, 2)
    root.diffusion_model.linear = foreign

    with pytest.raises(RuntimeError, match="lost its native operation identity"):
        with scope.activate():
            raise AssertionError("stale scope must not enter")

    assert root.diffusion_model.linear is foreign


def test_scope_fails_before_mutation_when_native_forward_changed() -> None:
    """Reject a foreign forward without altering native registration."""

    root, original, replacement = _graph()
    scope = RegionalOperationCallScope(
        root,
        (RegionalCallScopedOperation("diffusion_model.linear", original, replacement),),
    )

    def foreign_forward(inputs: torch.Tensor) -> torch.Tensor:
        """Return an externally replaced result."""

        return inputs + 5.0

    object.__setattr__(original, "forward", foreign_forward)
    with pytest.raises(RuntimeError, match="lost its native forward identity"):
        with scope.activate():
            raise AssertionError("stale scope must not enter")

    assert original.__dict__["forward"] is foreign_forward


def _graph() -> tuple[_RootGraph, nn.Module, nn.Module]:
    """Return one nested native graph and distinct lightweight replacement."""

    original = nn.Linear(2, 2)
    replacement = nn.Identity()
    diffusion = _DiffusionGraph(original)
    root = _RootGraph(diffusion)
    return root, original, replacement
