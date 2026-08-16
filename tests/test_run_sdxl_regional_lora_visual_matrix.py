# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the thin U11 managed visual-matrix coordinator."""

from __future__ import annotations

from pathlib import Path

import pytest
from sdxl_visual_test_inventory import visual_inventory

import tools.sdxl_attention_coupling_integration.visual_matrix_execution as execution
from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    SdxlVisualCase,
    VisualMaskProfile,
)
from tools.sdxl_attention_coupling_integration.visual_case_selection import (
    select_visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_cases import (
    visual_cases,
)
from tools.sdxl_attention_coupling_integration.visual_results import (
    SdxlVisualResultRecorder,
)
from tools.sdxl_attention_coupling_integration.visual_workflow import (
    BuiltSdxlVisualWorkflow,
)


class _FakeLinks:
    """Model cleanup-owned Comfy model visibility."""

    cleaned = False

    def __enter__(self) -> _FakeLinks:
        """Expose all fake links."""

        self.cleaned = False
        return self

    def __exit__(self, *args: object) -> None:
        """Mark all fake links cleaned."""

        del args
        self.cleaned = True


class _FakeMasks:
    """Model all cleanup-owned mask profiles."""

    cleaned = False

    def __init__(self, **kwargs: object) -> None:
        """Accept coordinator mask construction inputs."""

        del kwargs

    def __enter__(self) -> _FakeMasks:
        """Expose fake masks."""

        self.cleaned = False
        return self

    def __exit__(self, *args: object) -> None:
        """Mark all fake masks cleaned."""

        del args
        self.cleaned = True

    def names(self, profile: VisualMaskProfile) -> tuple[str, str]:
        """Return profile-specific fake input names."""

        return f"{profile.value}-left.png", f"{profile.value}-right.png"

    def evidence(self) -> tuple[dict[str, object], ...]:
        """Return minimal stable mask evidence."""

        return ({"profile": "fake"},)


class _FakeProcess:
    """Expose post-context liveness."""

    is_running = True


class _FakeRunning:
    """Provide the ready state consumed by the coordinator."""

    process = _FakeProcess()
    system_stats: JsonObject = {"system": "ready"}
    port = 8299


class _FakeServer:
    """Model exact process cleanup around all case submissions."""

    def __init__(self, **kwargs: object) -> None:
        """Accept coordinator server construction inputs."""

        assert kwargs["launch_arguments"] == (
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes",
            "SimpleSyrup",
            "SimpleSyrupBenchmarkProbe",
            "substitute-backend",
        )

    def __enter__(self) -> _FakeRunning:
        """Expose one ready fake server."""

        _FakeRunning.process.is_running = True
        return _FakeRunning()

    def __exit__(self, *args: object) -> None:
        """Stop the exact fake process."""

        del args
        _FakeRunning.process.is_running = False


class _FakeRecorder:
    """Require all cases before complete external cleanup finalization."""

    case_ids: list[str] = []

    def __init__(self, root: Path, *, cases: tuple[SdxlVisualCase, ...]) -> None:
        """Retain one managed result root."""

        self._root = root
        self._cases = cases
        type(self).case_ids = []

    def finalize(
        self,
        *,
        system_stats: JsonObject,
        server_cleanup: bool,
        model_cleanup: bool,
        mask_cleanup: bool,
        mask_evidence: tuple[dict[str, object], ...],
    ) -> Path:
        """Return a result only after all cases and owners are complete."""

        assert self.case_ids == [case.case_id for case in self._cases]
        assert system_stats == {"system": "ready"}
        assert server_cleanup and model_cleanup and mask_cleanup
        assert mask_evidence == ({"profile": "fake"},)
        result = self._root / "u11-result.json"
        result.write_text('{"status":"completed"}\n', encoding="utf-8")
        return result


def test_runner_submits_every_case_and_finalizes_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one server trajectory around all cases and clean exact owners."""

    links = _FakeLinks()
    monkeypatch.setattr(
        execution,
        "build_sdxl_visual_model_links",
        lambda comfy_root, inventory: links,
    )
    monkeypatch.setattr(execution, "ManagedSdxlVisualMasks", _FakeMasks)
    monkeypatch.setattr(execution, "ManagedComfyServer", _FakeServer)
    monkeypatch.setattr(execution, "SdxlVisualResultRecorder", _FakeRecorder)
    monkeypatch.setattr(execution, "_execute_case", _record_case)
    rendered: list[Path] = []

    def render(path: Path) -> tuple[Path, ...]:
        """Record each incremental and final sheet refresh."""

        rendered.append(path)
        return ()

    monkeypatch.setattr(execution, "build_visual_review_sheets", render)
    monkeypatch.setattr(
        execution,
        "is_loopback_port_available",
        lambda port: port == 8299,
    )
    artifacts = IntegrationArtifacts(tmp_path)
    inventory = visual_inventory(tmp_path)
    cases = visual_cases(inventory)

    result = execution.execute_visual_cases(
        artifacts,
        inventory=inventory,
        cases=cases,
        comfy_root=Path("<COMFY_ROOT>"),
        readiness_timeout=10.0,
        prompt_timeout=12.0,
    )

    assert result.read_text(encoding="utf-8") == '{"status":"completed"}\n'
    assert rendered == [artifacts.root / "u11-result.json" for _case in cases] + [
        result
    ]


def test_visual_case_selection_preserves_order_and_rejects_invalid_ids(
    tmp_path: Path,
) -> None:
    """Keep focused managed runs explicit without changing full defaults."""

    available = visual_cases(visual_inventory(tmp_path))

    selected = select_visual_cases(
        available,
        ("regional-style", "different-characters"),
    )

    assert tuple(case.case_id for case in selected) == (
        "regional-style",
        "different-characters",
    )
    with pytest.raises(ValueError, match="unique"):
        select_visual_cases(available, ("regional-style", "regional-style"))
    with pytest.raises(ValueError, match="Unknown"):
        select_visual_cases(available, ("missing",))


def _record_case(
    workflow: BuiltSdxlVisualWorkflow,
    *,
    case: SdxlVisualCase,
    running: object,
    recorder: SdxlVisualResultRecorder,
    prompt_timeout: float,
) -> None:
    """Record case order at the already-tested execution boundary."""

    del workflow, running, recorder
    assert prompt_timeout == 12.0
    _FakeRecorder.case_ids.append(case.case_id)
