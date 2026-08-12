"""Verify mandatory target, ordering, schedule, and digest evidence."""

from __future__ import annotations

import copy

import pytest
from anima_lora_characterization_fixtures import completed_outputs, inventory

from tools.anima_lora_characterization.evidence_validation import (
    validate_run_evidence,
)
from tools.anima_lora_characterization.history_outputs import LoraCompletedOutputs
from tools.anima_lora_characterization.matrix import runs


@pytest.mark.parametrize(
    "profile_id,capture_outputs",
    [("static-0", False), ("static-4", True), ("scheduled-4", False)],
)
def test_complete_evidence_passes_for_required_profile_classes(
    profile_id: str, capture_outputs: bool
) -> None:
    """Accept zero, stacked static, stacked scheduled, and digest cases."""

    run = next(
        item
        for item in runs()
        if item.profile.profile_id == profile_id
        and item.capture_outputs is capture_outputs
    )

    validate_run_evidence(run, completed_outputs(run), inventory())


def test_validation_rejects_partial_target_surface() -> None:
    """Never accept a run that silently omits an admitted ADAPTER_A target."""

    run = next(item for item in runs() if item.profile.profile_id == "static-1")
    outputs = completed_outputs(run)
    changed = copy.deepcopy(outputs.metrics)
    static = changed["static_patches"]
    assert isinstance(static, dict)
    static["target_keys"] = []

    with pytest.raises(ValueError, match="targets"):
        validate_run_evidence(
            run, LoraCompletedOutputs(changed, outputs.image), inventory()
        )


def test_validation_rejects_missing_schedule_interval() -> None:
    """Require every observed WeightHook strength transition."""

    run = next(item for item in runs() if item.profile.profile_id == "scheduled-2")
    outputs = completed_outputs(run)
    changed = copy.deepcopy(outputs.metrics)
    schedules = changed["schedule_observations"]
    assert isinstance(schedules, list)
    schedules.pop()

    with pytest.raises(ValueError, match="transitions"):
        validate_run_evidence(
            run, LoraCompletedOutputs(changed, outputs.image), inventory()
        )
