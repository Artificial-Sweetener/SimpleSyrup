"""Verify the thin managed SDXL integration coordinator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import tools.run_sdxl_attention_coupling_integration as runner
from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_coupling_integration.history import (
    SdxlModeHistoryEvidence,
)
from tools.sdxl_attention_coupling_integration.matrix import MODES
from tools.sdxl_attention_coupling_integration.workflow import (
    BuiltSdxlAttentionCouplingWorkflow,
)


class _FakeCheckpoint:
    """Model an exact context-owned checkpoint link."""

    checkpoint_name = "simple_syrup_p8_5_sdxl\\checkpoint.safetensors"

    def __init__(self, **kwargs: object) -> None:
        """Accept the coordinator's checkpoint construction values."""

        del kwargs
        self.cleaned = False

    def __enter__(self) -> _FakeCheckpoint:
        """Expose the owned stable Comfy checkpoint name."""

        return self

    def __exit__(self, *args: object) -> None:
        """Mark only the owned link cleaned."""

        del args
        self.cleaned = True


class _FakeMasks:
    """Model exact context-owned split-mask artifacts."""

    names = ("left.png", "right.png")

    def __init__(self, **kwargs: object) -> None:
        """Accept the coordinator's mask construction values."""

        del kwargs
        self.cleaned = False

    def __enter__(self) -> _FakeMasks:
        """Expose the owned mask names."""

        return self

    def __exit__(self, *args: object) -> None:
        """Mark both owned masks cleaned."""

        del args
        self.cleaned = True

    def evidence(self) -> tuple[dict[str, object], ...]:
        """Return stable mask evidence for persistence."""

        return ({"name": self.names[0]}, {"name": self.names[1]})


class _FakeProcess:
    """Expose post-context liveness to cleanup verification."""

    is_running = True


class _FakeClient:
    """Implement the coordinator's existing Comfy HTTP boundary."""

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Accept a graph containing every public sampler mode."""

        classes = {node["class_type"] for node in prompt.values()}
        assert {mode.node_id for mode in MODES}.issubset(classes)
        return "prompt-id"

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Return one opaque completed history value for the decoder boundary."""

        assert prompt_id == "prompt-id"
        assert timeout == 12.0
        return {"status": {"completed": True, "status_str": "success"}}

    def download_image(self, reference: ImageReference) -> bytes:
        """Return mode-labeled bytes for one decoded image reference."""

        return reference.filename.encode("utf-8")


class _FakeRunning:
    """Provide the ready server state consumed by the coordinator."""

    client = _FakeClient()
    process = _FakeProcess()
    system_stats: JsonObject = {"system": "ready"}
    port = 8299


class _FakeManagedServer:
    """Model guaranteed process cleanup at the context boundary."""

    def __init__(self, **kwargs: object) -> None:
        """Accept and retain no lower-level lifecycle policy."""

        del kwargs

    def __enter__(self) -> _FakeRunning:
        """Return one ready managed-server value."""

        _FakeRunning.process.is_running = True
        return _FakeRunning()

    def __exit__(self, *args: object) -> None:
        """Stop the exact fake process."""

        del args
        _FakeRunning.process.is_running = False


class _FakeRecorder:
    """Assert finalization occurs only after every external owner cleans up."""

    def __init__(self, root: Path) -> None:
        """Retain the managed artifact root."""

        self._root = root
        self._recorded = False

    def record_workflow(
        self,
        workflow: BuiltSdxlAttentionCouplingWorkflow,
        *,
        history: JsonObject,
        prompt_id: str,
        evidence: dict[str, SdxlModeHistoryEvidence],
        image_bytes: dict[str, bytes],
        masks: tuple[dict[str, object], ...],
        wall_runtime_ms: float,
    ) -> None:
        """Verify ordered complete evidence reaches the persistence boundary."""

        del workflow, history
        expected = tuple(mode.mode_id for mode in MODES)
        assert prompt_id == "prompt-id"
        assert tuple(evidence) == expected
        assert tuple(image_bytes) == expected
        assert tuple(mask["name"] for mask in masks) == ("left.png", "right.png")
        assert wall_runtime_ms > 0
        self._recorded = True

    def finalize(
        self,
        *,
        system_stats: JsonObject,
        server_cleanup: bool,
        checkpoint_cleanup: bool,
        mask_cleanup: bool,
    ) -> Path:
        """Return the durable result only after complete cleanup proof."""

        assert self._recorded
        assert system_stats == {"system": "ready"}
        assert server_cleanup and checkpoint_cleanup and mask_cleanup
        result = self._root / "result.json"
        result.write_text('{"status": "completed"}\n', encoding="utf-8")
        return result


def test_execute_sdxl_matrix_records_only_after_all_owned_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinate all three modes and finalize after link, masks, and server."""

    monkeypatch.setattr(runner, "ManagedCheckpointLink", _FakeCheckpoint)
    monkeypatch.setattr(runner, "ManagedSdxlSplitMasks", _FakeMasks)
    monkeypatch.setattr(runner, "ManagedComfyServer", _FakeManagedServer)
    monkeypatch.setattr(runner, "SdxlIntegrationResultRecorder", _FakeRecorder)
    monkeypatch.setattr(runner, "decode_sdxl_history", _decode_evidence)
    monkeypatch.setattr(runner, "is_loopback_port_available", lambda port: port == 8299)
    artifacts = IntegrationArtifacts(tmp_path)

    result = runner.execute_sdxl_matrix(
        artifacts,
        comfy_root=Path("<COMFY_ROOT>"),
        checkpoint_path=Path("E:/models/checkpoint.safetensors"),
        readiness_timeout=10.0,
        prompt_timeout=12.0,
    )

    assert result.read_text(encoding="utf-8") == '{"status": "completed"}\n'
    decoded: object = json.loads((artifacts.root / "run.json").read_text("utf-8"))
    assert isinstance(decoded, dict)
    record = cast(dict[str, object], decoded)
    assert record["status"] == "completed"


def _decode_evidence(
    history: JsonObject,
    workflow: BuiltSdxlAttentionCouplingWorkflow,
) -> dict[str, SdxlModeHistoryEvidence]:
    """Return complete mode evidence at the already-tested decoder boundary."""

    del history, workflow
    return {
        mode.mode_id: SdxlModeHistoryEvidence(
            ImageReference(f"{mode.mode_id}.png", "p8.5", "output"),
            {"model_call_count": mode.expected_model_calls},
            {"snapshots": []},
        )
        for mode in MODES
    }
