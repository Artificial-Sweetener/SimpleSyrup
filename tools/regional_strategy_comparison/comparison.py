"""Validate P10.2 cross-strategy and regional-LoRA execution invariants."""

from __future__ import annotations

from dataclasses import dataclass

from tools.comfy_api import JsonObject

from .history import StrategyProbeMetrics
from .matrix import StrategyComparisonCase, cases, expected_low_rank_multiplier


@dataclass(frozen=True, slots=True)
class StrategyComparisonObservation:
    """Expose one matrix case's managed execution evidence."""

    case: StrategyComparisonCase
    metrics: StrategyProbeMetrics
    diagnostics: tuple[JsonObject, ...]


class StrategyComparisonValidator:
    """Prove matched call geometry and single-trajectory Attention Coupling."""

    def validate(
        self,
        observations: tuple[StrategyComparisonObservation, ...],
    ) -> None:
        """Reject incomplete, mismatched, or duplicated execution evidence."""

        expected_ids = tuple(case.case_id for case in cases())
        observed_ids = tuple(observation.case.case_id for observation in observations)
        if observed_ids != expected_ids:
            raise ValueError(
                "P10.2 observations must match the complete ordered matrix."
            )
        by_id = {observation.case.case_id: observation for observation in observations}
        if len(by_id) != len(observations):
            raise ValueError("P10.2 observations contain duplicate case identities.")
        by_profile: dict[str, list[StrategyComparisonObservation]] = {}
        for observation in observations:
            by_profile.setdefault(observation.case.spatial_profile, []).append(
                observation
            )
            self._validate_case(observation)
        for profile, profile_observations in by_profile.items():
            self._validate_profile(profile, profile_observations)

    @staticmethod
    def _validate_case(observation: StrategyComparisonObservation) -> None:
        """Validate one strategy's exact diagnostic contract."""

        case = observation.case
        metrics = observation.metrics
        if case.strategy == "regional_conditioning":
            if observation.diagnostics:
                raise ValueError(
                    "P10.2 legacy control emitted Attention Coupling diagnostics."
                )
            return
        if len(observation.diagnostics) != metrics.model_call_count:
            raise ValueError(
                f"P10.2 {case.case_id} must emit one diagnostic per model call."
            )
        expected_low_rank = expected_low_rank_multiplier(case)
        for snapshot in observation.diagnostics:
            if snapshot.get("strategy") != "attention_coupling":
                raise ValueError("P10.2 diagnostic strategy identity is incorrect.")
            work = snapshot.get("estimated_work")
            if not isinstance(work, dict):
                raise TypeError("P10.2 diagnostic estimated_work must be an object.")
            if work.get("denoiser_call_multiplier") != 1.0:
                raise ValueError("P10.2 diagnostic duplicated the denoiser trajectory.")
            if work.get("cross_attention_branch_multiplier") != 3.0:
                raise ValueError("P10.2 diagnostic branch multiplier must equal three.")
            observed_low_rank = work.get("low_rank_adapter_multiplier", 0.0)
            if observed_low_rank != expected_low_rank:
                raise ValueError(
                    f"P10.2 {case.case_id} low-rank multiplier must equal "
                    f"{expected_low_rank}."
                )

    @staticmethod
    def _validate_profile(
        profile: str,
        observations: list[StrategyComparisonObservation],
    ) -> None:
        """Compare no-LoRA strategies and Attention Coupling LoRA scaling."""

        regional = next(
            (
                item
                for item in observations
                if item.case.strategy == "regional_conditioning"
            ),
            None,
        )
        attention = sorted(
            (
                item
                for item in observations
                if item.case.strategy == "attention_coupling"
            ),
            key=lambda item: item.case.regional_lora_count,
        )
        if regional is None or not attention or attention[0].case.regional_lora_count:
            raise ValueError(f"P10.2 {profile} is missing matched no-LoRA controls.")
        control = attention[0]
        if regional.metrics.model_call_count != control.metrics.model_call_count:
            raise ValueError(f"P10.2 {profile} strategies changed model-call count.")
        expected_regional_sizes = tuple(
            size * 2 for size in control.metrics.model_input_batch_sizes
        )
        if regional.metrics.model_input_batch_sizes != expected_regional_sizes:
            raise ValueError(
                f"P10.2 {profile} Regional Conditioning must execute two complete "
                "transformer trajectories per Attention Coupling trajectory."
            )
        for scaled in attention[1:]:
            if scaled.metrics.model_call_count != control.metrics.model_call_count:
                raise ValueError(f"P10.2 {profile} regional LoRAs added model calls.")
            if (
                scaled.metrics.model_input_batch_sizes
                != control.metrics.model_input_batch_sizes
            ):
                raise ValueError(
                    f"P10.2 {profile} regional LoRAs changed trajectory batch geometry."
                )
