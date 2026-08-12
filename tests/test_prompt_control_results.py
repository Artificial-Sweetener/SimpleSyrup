"""Verify durable Prompt Control result completeness and resume identity."""

from pathlib import Path

import pytest
from test_prompt_control_evidence_validation import _outputs

from tools.prompt_control_characterization.cases import cases
from tools.prompt_control_characterization.results import PromptControlResultRecorder
from tools.prompt_control_characterization.source_identity import (
    PromptControlSourceIdentity,
)


def test_recorder_finalizes_only_complete_successful_matrix(tmp_path: Path) -> None:
    """Persist one custom matrix in authoritative order and finalize it."""

    case = cases()[0]
    recorder = PromptControlResultRecorder(
        tmp_path,
        PromptControlSourceIdentity("test", 1, "a" * 64),
        {"runtime": "test"},
        (case,),
    )
    with pytest.raises(ValueError, match="incomplete"):
        recorder.finalize()
    recorder.record_success(case, _outputs(case), {"1": {"class_type": "Test"}})
    result = recorder.finalize()
    assert result.is_file()
    assert case.case_id in recorder.completed_case_ids


def test_recorder_retains_failed_case_as_incomplete(tmp_path: Path) -> None:
    """Never omit or qualify a failed required matrix position."""

    case = cases()[0]
    recorder = PromptControlResultRecorder(
        tmp_path,
        PromptControlSourceIdentity("test", 1, "b" * 64),
        {},
        (case,),
    )
    recorder.record_failure(case, RuntimeError("failure"))
    with pytest.raises(ValueError, match="failed"):
        recorder.finalize()
