# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify copy-on-write persistent standard-UNet diffusion shells."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from simple_syrup.runtime.regional_lora.standard_unet_variant_materialization import (
    StandardUnetMaterializedVariant,
    StandardUnetVariantParameter,
)
from simple_syrup.runtime.regional_lora.standard_unet_variant_shell import (
    StandardUnetVariantShellBuilder,
)


class _Branch(nn.Module):
    """Expose targeted and shared leaves through one ancestor."""

    def __init__(self) -> None:
        """Create deterministic generic projection leaves."""

        super().__init__()
        self.target = nn.Linear(2, 2, bias=False)
        self.shared = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run both leaves."""

        return cast(torch.Tensor, self.target(inputs) + self.shared(inputs))


class _Diffusion(nn.Module):
    """Expose one target-bearing branch and an unrelated leaf."""

    def __init__(self) -> None:
        """Create the generic source graph."""

        super().__init__()
        self.branch = _Branch()
        self.unrelated = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the target-bearing branch."""

        return cast(torch.Tensor, self.unrelated(self.branch(inputs)))


def test_shell_replaces_only_target_parameter_and_preserves_source() -> None:
    """Copy target ancestors while sharing every untouched module object."""

    source = _Diffusion()
    source.branch.target.weight.data.fill_(1.0)
    source.branch.shared.weight.data.zero_()
    materialized = StandardUnetMaterializedVariant(
        0,
        (
            StandardUnetVariantParameter(
                "branch.target.weight",
                torch.full_like(source.branch.target.weight, 3.0).detach(),
            ),
        ),
    )

    shell = StandardUnetVariantShellBuilder().build(source, materialized)

    assert isinstance(shell, _Diffusion)
    assert shell is not source
    assert shell.branch is not source.branch
    assert shell.branch.target is not source.branch.target
    assert shell.branch.shared is source.branch.shared
    assert shell.unrelated is source.unrelated
    assert shell.branch.target.weight is not source.branch.target.weight
    assert torch.equal(source.branch.target.weight, torch.ones((2, 2)))
    assert torch.equal(shell(torch.ones((1, 2))), torch.full((1, 2), 6.0))


def test_shell_reports_every_missing_target_path() -> None:
    """Preserve complete canonical evidence when target branches are absent."""

    source = _Diffusion()
    materialized = StandardUnetMaterializedVariant(
        0,
        (
            StandardUnetVariantParameter(
                "absent.weight",
                torch.ones_like(source.branch.target.weight),
            ),
            StandardUnetVariantParameter(
                "branch.absent.weight",
                torch.ones_like(source.branch.target.weight),
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Variant shell did not bind target paths "
            "\\('absent.weight', 'branch.absent.weight'\\)"
        ),
    ):
        StandardUnetVariantShellBuilder().build(source, materialized)


def test_shell_rejects_null_and_incompatible_parameters() -> None:
    """Preserve fail-closed parameter and tensor compatibility contracts."""

    source = _Diffusion()
    null_parameter = StandardUnetMaterializedVariant(
        0,
        (
            StandardUnetVariantParameter(
                "branch.target.bias",
                torch.ones(2),
            ),
        ),
    )
    with pytest.raises(TypeError, match="must be a Parameter"):
        StandardUnetVariantShellBuilder().build(source, null_parameter)

    wrong_shape = StandardUnetMaterializedVariant(
        0,
        (
            StandardUnetVariantParameter(
                "branch.target.weight",
                torch.ones(3, 2),
            ),
        ),
    )
    with pytest.raises(ValueError, match="is incompatible"):
        StandardUnetVariantShellBuilder().build(source, wrong_shape)


def test_shell_binds_pre_resident_target_device() -> None:
    """Accept exact variant tensors already placed for Comfy model residency."""

    source = _Diffusion()
    materialized = StandardUnetMaterializedVariant(
        0,
        (
            StandardUnetVariantParameter(
                "branch.target.weight",
                torch.empty_like(source.branch.target.weight, device="meta"),
            ),
        ),
    )

    shell = StandardUnetVariantShellBuilder().build(source, materialized)

    assert isinstance(shell, _Diffusion)
    assert shell.branch.target.weight.device == torch.device("meta")
    assert source.branch.target.weight.device == torch.device("cpu")
