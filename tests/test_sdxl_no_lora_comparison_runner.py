# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Characterize the matched no-LoRA comparison lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

from pytest import MonkeyPatch
from sdxl_visual_test_inventory import visual_inventory

import tools.sdxl_no_lora_comparison.runner as runner
from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)


class _FakeClient:
    """Record submitted graphs without invoking ComfyUI."""

    def __init__(self) -> None:
        """Initialize the submitted-prompt evidence collection."""

        self.prompts: list[dict[str, JsonObject]] = []

    def submit(self, prompt: dict[str, JsonObject]) -> str:
        """Capture one workflow and return a stable prompt identity."""

        self.prompts.append(prompt)
        return f"prompt-{len(self.prompts)}"

    def wait_for_history(self, prompt_id: str, *, timeout: float) -> JsonObject:
        """Accept the locked managed timeout for every submitted graph."""

        assert prompt_id.startswith("prompt-")
        assert timeout == 30.0
        return {}


class _WorkflowWithPrompt(Protocol):
    """Describe the focused workflow surface observed by the fake client."""

    prompt: dict[str, JsonObject]


class _FakeManagedServer:
    """Provide one stopped managed-server surface for lifecycle verification."""

    client = _FakeClient()

    def __init__(self, **values: object) -> None:
        """Retain the required graph-node set for the isolated process."""

        required = values["required_node_ids"]
        assert isinstance(required, frozenset)
        assert "KSampler" in required
        assert "SimpleSyrup.KSamplerAttentionCoupling" in required
        self.running = SimpleNamespace(
            system_stats={"device": "test"},
            client=type(self).client,
            port=54321,
            process=SimpleNamespace(is_running=False),
        )

    def __enter__(self) -> object:
        """Return the controlled managed runtime."""

        return self.running

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Leave the controlled process stopped."""

        del exc_type, exc, traceback


class _FakeModelLinks:
    """Model the checkpoint-link lifecycle without filesystem mutation."""

    def __init__(self, **values: object) -> None:
        """Accept the focused checkpoint-only model visibility declaration."""

        links = values["links"]
        assert isinstance(links, tuple)
        assert len(links) == 1
        self.cleaned = True

    def __enter__(self) -> _FakeModelLinks:
        """Expose the active managed model links."""

        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Leave the fake model-link owner cleaned."""

        del exc_type, exc, traceback


class _FakeMasks:
    """Supply stable hard-mask names and cleanup confirmation."""

    cleaned = True

    def __init__(self, **values: object) -> None:
        """Accept the generated run identity and input root."""

        assert isinstance(values["input_root"], Path)
        assert isinstance(values["run_id"], str)

    def __enter__(self) -> _FakeMasks:
        """Expose the mask owner for graph construction."""

        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Leave the fake mask owner cleaned."""

        del exc_type, exc, traceback

    def names(self, _profile: object) -> tuple[str, str]:
        """Return the declared left and right hard-mask names."""

        return ("left.png", "right.png")


class _FakeMeasurementRecorder:
    """Record measured workflow submission without image decoding."""

    def __init__(self, _artifacts: object) -> None:
        """Accept the artifact owner used by the real recorder."""

    def measure_plain(self, client: object, **values: object) -> JsonObject:
        """Submit the ordinary graph and return its timing evidence."""

        _submit_measured_workflow(client, values)
        return {"model_runtime_ms": 10.0}

    def measure_regional(self, client: object, **values: object) -> JsonObject:
        """Submit the regional graph and return its timing evidence."""

        _submit_measured_workflow(client, values)
        return {"model_runtime_ms": 20.0}


def test_runner_warms_both_paths_then_records_two_measured_outputs(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Keep the same-process order, warmups, and cleanup result explicit."""

    _FakeManagedServer.client = _FakeClient()
    monkeypatch.setattr(
        runner,
        "resolve_active_comfy_model_root",
        lambda _root: tmp_path,
    )
    monkeypatch.setattr(runner, "ManagedSdxlVisualModelLinks", _FakeModelLinks)
    monkeypatch.setattr(runner, "ManagedSdxlVisualMasks", _FakeMasks)
    monkeypatch.setattr(runner, "ManagedComfyServer", _FakeManagedServer)
    monkeypatch.setattr(runner, "is_loopback_port_available", lambda _port: True)
    monkeypatch.setattr(
        runner,
        "SdxlNoLoraMeasurementRecorder",
        _FakeMeasurementRecorder,
    )
    artifacts = IntegrationArtifacts(tmp_path / "artifacts")

    result = runner.run_no_lora_comparison(
        artifacts,
        inventory=visual_inventory(tmp_path),
        prompts=_prompts(),
        comfy_root=tmp_path,
        readiness_timeout=10.0,
        prompt_timeout=30.0,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    submitted = _FakeManagedServer.client.prompts
    assert len(submitted) == 4
    assert all("SaveImage" not in _types(prompt) for prompt in submitted[:2])
    assert all("VAEDecode" not in _types(prompt) for prompt in submitted[:2])
    assert _types(submitted[2]).count("SaveImage") == 1
    assert _types(submitted[3]).count("SaveImage") == 1
    assert all(":" not in _save_prefix(prompt) for prompt in submitted[2:])
    assert payload["regional_to_plain_model_ratio"] == 2.0
    assert payload["regional_model_overhead_percent"] == 100.0
    assert payload["cleanup"] == {
        "masks": True,
        "model_links": True,
        "port": 54321,
        "server": True,
    }


def _submit_measured_workflow(client: object, values: dict[str, object]) -> None:
    """Record one measured graph through the same fake client boundary."""

    assert isinstance(client, _FakeClient)
    workflow = cast(_WorkflowWithPrompt, values["workflow"])
    client.submit(workflow.prompt)


def _types(prompt: dict[str, JsonObject]) -> tuple[str, ...]:
    """Return exact node types declared by one submitted graph."""

    return tuple(str(node["class_type"]) for node in prompt.values())


def _save_prefix(prompt: dict[str, JsonObject]) -> str:
    """Return one measured workflow's sole SaveImage prefix."""

    node = next(
        value for value in prompt.values() if value["class_type"] == "SaveImage"
    )
    inputs = node["inputs"]
    assert isinstance(inputs, dict)
    prefix = inputs["filename_prefix"]
    assert isinstance(prefix, str)
    return prefix


def _prompts() -> SdxlVisualPromptSet:
    """Return external-like pink/black section prompts without fixture identity."""

    return SdxlVisualPromptSet(
        base_positive_g="global positive",
        base_positive_l="global positive",
        base_negative_g="global negative",
        base_negative_l="global negative",
        left_positive_g="pink-haired left subject",
        left_positive_l="pink-haired left subject",
        right_positive_g="black-haired right subject",
        right_positive_l="black-haired right subject",
        left_negative_g="left negative",
        left_negative_l="left negative",
        right_negative_g="right negative",
        right_negative_l="right negative",
    )
