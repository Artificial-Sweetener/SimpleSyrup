"""Define backend-neutral Attention Coupling invariant test values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from simple_syrup.domain.processed_regional_attention import (
    ProcessedRegionalAttentionPlan,
)
from simple_syrup.domain.regional_attention import RegionalAttentionBranch


@dataclass(frozen=True, slots=True)
class AttentionCouplingInvariantEntry:
    """Describe one regional context entry without backend patch details."""

    batch_values: tuple[float, ...]
    strengths: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AttentionCouplingInvariantScenario:
    """Describe canonical inputs and explicit backend numerical expectations."""

    masks: torch.Tensor
    base_values: tuple[float, ...]
    regions: tuple[tuple[AttentionCouplingInvariantEntry, ...], ...]
    region_strengths: tuple[float, ...]
    expected: torch.Tensor
    anima_expected: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class AttentionCouplingInvariantObservation:
    """Publish only behavior shared by supported attention backends."""

    output: torch.Tensor
    native_attention_calls: int
    packed_batch_size: int
    packed_context_values: tuple[float, ...]
    query_unchanged: bool
    contexts_unchanged: bool
    masks_unchanged: bool


@dataclass(frozen=True, slots=True)
class AttentionCouplingCallScenario:
    """Describe one backend-neutral scheduled CFG model call."""

    plan: ProcessedRegionalAttentionPlan
    selectors: tuple[int, ...]
    sigma: float


@dataclass(frozen=True, slots=True)
class AttentionCouplingCallObservation:
    """Publish shared scheduled CFG state observed inside one model call."""

    branches: tuple[RegionalAttentionBranch, ...]
    base_values: tuple[float, ...]
    regional_entry_values: tuple[tuple[float, ...], ...]
    nested_model_calls: int
    source_context_unchanged: bool
    state_restored: bool
    canonical_plan_preserved: bool


class AttentionCouplingInvariantHarness(Protocol):
    """Execute shared scenarios behind one model-specific test adapter."""

    def execute(
        self,
        scenario: AttentionCouplingInvariantScenario,
    ) -> AttentionCouplingInvariantObservation:
        """Return backend-neutral observations for one complete attention call."""

    def validate_masks(self, masks: torch.Tensor) -> None:
        """Validate canonical masks through the backend execution boundary."""

    def expected_output(
        self,
        scenario: AttentionCouplingInvariantScenario,
    ) -> torch.Tensor:
        """Return the declared numerical expectation for this backend."""


class AttentionCouplingCallHarness(Protocol):
    """Resolve shared schedule and CFG behavior through one backend wrapper."""

    def resolve_call(
        self,
        scenario: AttentionCouplingCallScenario,
    ) -> AttentionCouplingCallObservation:
        """Resolve schedule and CFG state through the backend call wrapper."""


@dataclass(frozen=True, slots=True)
class AttentionCouplingLifecycleObservation:
    """Publish clone lifecycle facts shared by supported backends."""

    derived_is_new: bool
    direct_parent: bool
    source_unchanged: bool
    derived_has_mutations: bool
    canonical_plan_preserved: bool


class AttentionCouplingLifecycleHarness(Protocol):
    """Derive one backend model behind shared lifecycle observations."""

    def derive(self) -> AttentionCouplingLifecycleObservation:
        """Return one clone lifecycle observation from the production backend."""


@dataclass(frozen=True, slots=True)
class AttentionCouplingDiagnosticsObservation:
    """Publish JSON-safe diagnostic facts shared by supported backends."""

    strategy: str
    backend_identity: str
    runtime_identity: str
    region_count: int
    active_region_indices: tuple[int, ...]
    active_branches: tuple[str, ...]
    cross_attention_branch_multiplier: float
    denoiser_call_multiplier: float
    query_token_count: int


class AttentionCouplingDiagnosticsHarness(Protocol):
    """Build one shared diagnostic snapshot with backend-owned identity."""

    def observe(self) -> AttentionCouplingDiagnosticsObservation:
        """Return backend-neutral diagnostic values for one model call."""
