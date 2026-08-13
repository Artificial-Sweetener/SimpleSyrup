# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Verify the thin U11 managed visual-matrix coordinator."""

from __future__ import annotations

from pathlib import Path

import pytest

import tools.run_sdxl_regional_lora_visual_matrix as runner
from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_coupling_integration.visual_cases import (
    SdxlVisualCase,
    VisualMaskProfile,
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

    def __init__(self, root: Path) -> None:
        """Retain one managed result root."""

        self._root = root
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

        assert self.case_ids == [case.case_id for case in visual_cases()]
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
    monkeypatch.setattr(runner, "_model_links", lambda comfy_root: links)
    monkeypatch.setattr(runner, "ManagedSdxlVisualMasks", _FakeMasks)
    monkeypatch.setattr(runner, "ManagedComfyServer", _FakeServer)
    monkeypatch.setattr(runner, "SdxlVisualResultRecorder", _FakeRecorder)
    monkeypatch.setattr(runner, "_execute_case", _record_case)
    rendered: list[Path] = []

    def render(path: Path) -> tuple[Path, ...]:
        """Record each incremental and final sheet refresh."""

        rendered.append(path)
        return ()

    monkeypatch.setattr(runner, "build_visual_review_sheets", render)
    monkeypatch.setattr(runner, "is_loopback_port_available", lambda port: port == 8299)
    artifacts = IntegrationArtifacts(tmp_path)

    result = runner.execute_visual_matrix(
        artifacts,
        comfy_root=Path("<COMFY_ROOT>"),
        readiness_timeout=10.0,
        prompt_timeout=12.0,
    )

    assert result.read_text(encoding="utf-8") == '{"status":"completed"}\n'
    assert rendered == [
        artifacts.root / "u11-result.json" for _case in visual_cases()
    ] + [result]


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
