"""Verify P10.2 cross-case execution acceptance."""

import pytest

from tools.comfy_api import JsonObject
from tools.regional_strategy_comparison.comparison import (
    StrategyComparisonObservation,
    StrategyComparisonValidator,
)
from tools.regional_strategy_comparison.history import StrategyProbeMetrics
from tools.regional_strategy_comparison.matrix import (
    cases,
    expected_low_rank_multiplier,
)


def _observations() -> tuple[StrategyComparisonObservation, ...]:
    """Build a complete valid synthetic managed matrix."""

    observations = []
    for case in cases():
        attention = case.strategy == "attention_coupling"
        sizes = (1, 1) if attention else (2, 2)
        metrics = StrategyProbeMetrics(10.0, 100, 2, sizes, sum(sizes))
        diagnostics: tuple[JsonObject, ...] = ()
        if attention:
            work: JsonObject = {
                "denoiser_call_multiplier": 1.0,
                "cross_attention_branch_multiplier": 3.0,
            }
            low_rank = expected_low_rank_multiplier(case)
            if low_rank:
                work["low_rank_adapter_multiplier"] = low_rank
            snapshot: JsonObject = {
                "strategy": "attention_coupling",
                "estimated_work": work,
            }
            diagnostics = (snapshot, snapshot)
        observations.append(StrategyComparisonObservation(case, metrics, diagnostics))
    return tuple(observations)


def test_validator_accepts_single_trajectory_and_exact_lora_scaling() -> None:
    """Accept matched calls, twofold legacy work, and invariant LoRA geometry."""

    StrategyComparisonValidator().validate(_observations())


def test_validator_rejects_regional_lora_trajectory_growth() -> None:
    """Reject a regional adapter case that changes complete model batch work."""

    observations = list(_observations())
    index = next(
        index
        for index, observation in enumerate(observations)
        if observation.case.regional_lora_count == 1
    )
    original = observations[index]
    observations[index] = StrategyComparisonObservation(
        original.case,
        StrategyProbeMetrics(10.0, 100, 2, (2, 2), 4),
        original.diagnostics,
    )

    with pytest.raises(ValueError, match="trajectory batch geometry"):
        StrategyComparisonValidator().validate(tuple(observations))
