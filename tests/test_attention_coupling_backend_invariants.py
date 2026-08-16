# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run shared Attention Coupling invariants against every backend harness."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from anima_attention_coupling_call_harness import AnimaAttentionCouplingCallHarness
from anima_attention_coupling_diagnostics_harness import (
    AnimaAttentionCouplingDiagnosticsHarness,
)
from anima_attention_coupling_invariant_harness import (
    AnimaAttentionCouplingInvariantHarness,
)
from anima_attention_coupling_lifecycle_harness import (
    AnimaAttentionCouplingLifecycleHarness,
)
from attention_coupling_invariant_contract import (
    AttentionCouplingCallHarness,
    AttentionCouplingCallScenario,
    AttentionCouplingDiagnosticsHarness,
    AttentionCouplingInvariantEntry,
    AttentionCouplingInvariantHarness,
    AttentionCouplingInvariantScenario,
    AttentionCouplingLifecycleHarness,
)
from attention_coupling_invariant_values import scheduled_invariant_plan
from unet_attention_coupling_call_harness import UnetAttentionCouplingCallHarness
from unet_attention_coupling_diagnostics_harness import (
    UnetAttentionCouplingDiagnosticsHarness,
)
from unet_attention_coupling_invariant_harness import (
    UnetAttentionCouplingInvariantHarness,
)
from unet_attention_coupling_lifecycle_harness import (
    UnetAttentionCouplingLifecycleHarness,
)

from simple_syrup.domain.regional_attention import RegionalAttentionBranch


def _single_entries(
    *region_batch_values: tuple[float, ...],
) -> tuple[tuple[AttentionCouplingInvariantEntry, ...], ...]:
    """Return one unit-strength entry per canonical region."""

    return tuple(
        (
            AttentionCouplingInvariantEntry(
                batch_values,
                (1.0,) * len(batch_values),
            ),
        )
        for batch_values in region_batch_values
    )


@pytest.fixture(
    params=(
        AnimaAttentionCouplingInvariantHarness,
        UnetAttentionCouplingInvariantHarness,
    ),
    ids=("anima", "standard-unet"),
)
def backend_harness(
    request: pytest.FixtureRequest,
) -> AttentionCouplingInvariantHarness:
    """Return one model-specific adapter behind the shared behavior contract."""

    harness_type = cast(type[AttentionCouplingInvariantHarness], request.param)
    return harness_type()


@pytest.fixture(
    params=(
        AnimaAttentionCouplingCallHarness,
        UnetAttentionCouplingCallHarness,
    ),
    ids=("anima", "standard-unet"),
)
def call_harness(request: pytest.FixtureRequest) -> AttentionCouplingCallHarness:
    """Return one model-call adapter behind the shared behavior contract."""

    harness_type = cast(type[AttentionCouplingCallHarness], request.param)
    return harness_type()


@pytest.fixture(
    params=(
        AnimaAttentionCouplingLifecycleHarness,
        UnetAttentionCouplingLifecycleHarness,
    ),
    ids=("anima", "standard-unet"),
)
def lifecycle_harness(
    request: pytest.FixtureRequest,
) -> AttentionCouplingLifecycleHarness:
    """Return one derivation adapter behind the shared lifecycle contract."""

    harness_type = cast(type[AttentionCouplingLifecycleHarness], request.param)
    return harness_type()


@pytest.fixture(
    params=(
        AnimaAttentionCouplingDiagnosticsHarness,
        UnetAttentionCouplingDiagnosticsHarness,
    ),
    ids=("anima", "standard-unet"),
)
def diagnostics_harness(
    request: pytest.FixtureRequest,
) -> AttentionCouplingDiagnosticsHarness:
    """Return one backend identity adapter behind the diagnostic contract."""

    harness_type = cast(type[AttentionCouplingDiagnosticsHarness], request.param)
    return harness_type()


@pytest.mark.parametrize(
    "scenario",
    (
        AttentionCouplingInvariantScenario(
            masks=torch.zeros(2, 1, 2),
            base_values=(10.0, 20.0),
            regions=_single_entries((30.0, 40.0), (70.0, 80.0)),
            region_strengths=(1.0, 1.0),
            expected=torch.tensor([[[10.0], [10.0]], [[20.0], [20.0]]]),
        ),
        AttentionCouplingInvariantScenario(
            masks=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
            base_values=(10.0, 20.0),
            regions=_single_entries((30.0, 40.0), (70.0, 80.0)),
            region_strengths=(1.0, 1.0),
            expected=torch.tensor([[[30.0], [70.0]], [[40.0], [80.0]]]),
            anima_expected=torch.tensor(
                [[[70.0 / 3.0], [50.0]], [[100.0 / 3.0], [60.0]]]
            ),
        ),
        AttentionCouplingInvariantScenario(
            masks=torch.ones(2, 1, 1),
            base_values=(10.0, 20.0),
            regions=_single_entries((30.0, 40.0), (70.0, 80.0)),
            region_strengths=(1.0, 1.0),
            expected=torch.tensor([[[50.0]], [[60.0]]]),
            anima_expected=torch.tensor([[[110.0 / 3.0]], [[140.0 / 3.0]]]),
        ),
        AttentionCouplingInvariantScenario(
            masks=torch.tensor([[[0.25]], [[0.0]]]),
            base_values=(10.0, 20.0),
            regions=_single_entries((30.0, 40.0), (70.0, 80.0)),
            region_strengths=(1.0, 1.0),
            expected=torch.tensor([[[30.0]], [[40.0]]]),
            anima_expected=torch.tensor([[[40.0 / 3.0]], [[70.0 / 3.0]]]),
        ),
        AttentionCouplingInvariantScenario(
            masks=torch.ones(1, 1, 1),
            base_values=(10.0,),
            regions=(
                (
                    AttentionCouplingInvariantEntry((2.0,), (0.25,)),
                    AttentionCouplingInvariantEntry((6.0,), (0.75,)),
                ),
            ),
            region_strengths=(1.0,),
            expected=torch.tensor([[[5.0]]]),
            anima_expected=torch.tensor([[[20.0 / 3.0]]]),
        ),
        AttentionCouplingInvariantScenario(
            masks=torch.ones(1, 1, 1),
            base_values=(10.0,),
            regions=_single_entries((30.0,)),
            region_strengths=(0.0,),
            expected=torch.tensor([[[10.0]]]),
        ),
    ),
    ids=(
        "uncovered",
        "hard-partition-cfg-batch",
        "overlap-normalization",
        "soft-feather",
        "multiple-entries",
        "zero-strength",
    ),
)
def test_backends_preserve_declared_numerical_and_single_trajectory_invariants(
    backend_harness: AttentionCouplingInvariantHarness,
    scenario: AttentionCouplingInvariantScenario,
) -> None:
    """Match backend policy, immutability, determinism, and one native call."""

    first = backend_harness.execute(scenario)
    second = backend_harness.execute(scenario)

    torch.testing.assert_close(first.output, backend_harness.expected_output(scenario))
    torch.testing.assert_close(second.output, first.output)
    assert first.native_attention_calls == 1
    assert second.native_attention_calls == 1
    assert first.query_unchanged
    assert first.contexts_unchanged
    assert first.masks_unchanged


@pytest.mark.parametrize(
    ("masks", "message"),
    (
        (torch.full((1, 1, 1), float("nan")), "finite"),
        (torch.ones(1, 0, 1), "non-empty|positive"),
    ),
)
def test_backends_fail_closed_on_malformed_shared_masks(
    backend_harness: AttentionCouplingInvariantHarness,
    masks: torch.Tensor,
    message: str,
) -> None:
    """Reject malformed canonical masks before any attention call."""

    with pytest.raises((TypeError, ValueError), match=message):
        backend_harness.validate_masks(masks)


def test_backends_share_schedule_cfg_plan_and_model_call_invariants(
    call_harness: AttentionCouplingCallHarness,
) -> None:
    """Resolve one reversed-CFG scheduled call without duplicating model work."""

    scenario = AttentionCouplingCallScenario(
        plan=scheduled_invariant_plan(),
        selectors=(1, 0),
        sigma=0.5,
    )

    observation = call_harness.resolve_call(scenario)

    assert observation.branches == (
        RegionalAttentionBranch.NEGATIVE,
        RegionalAttentionBranch.POSITIVE,
    )
    assert observation.base_values == (-1.0, 1.0)
    assert observation.regional_entry_values == ((-2.0, 2.0),)
    assert observation.nested_model_calls == 1
    assert observation.source_context_unchanged
    assert observation.state_restored
    assert observation.canonical_plan_preserved


def test_backends_share_direct_clone_and_source_immutability_invariants(
    lifecycle_harness: AttentionCouplingLifecycleHarness,
) -> None:
    """Derive one mutated direct child without changing its source MODEL."""

    observation = lifecycle_harness.derive()

    assert observation.derived_is_new
    assert observation.direct_parent
    assert observation.source_unchanged
    assert observation.derived_has_mutations
    assert observation.canonical_plan_preserved


def test_backends_share_diagnostic_identity_selection_and_call_invariants(
    diagnostics_harness: AttentionCouplingDiagnosticsHarness,
) -> None:
    """Publish exact backend identity with common selection and call counts."""

    observation = diagnostics_harness.observe()

    assert observation.strategy == "attention_coupling"
    assert observation.backend_identity == observation.runtime_identity
    assert observation.region_count == 1
    assert observation.active_region_indices == (0,)
    assert observation.active_branches == ("negative", "positive")
    assert observation.cross_attention_branch_multiplier == 2.0
    assert observation.denoiser_call_multiplier == 1.0
    assert observation.query_token_count == 1
