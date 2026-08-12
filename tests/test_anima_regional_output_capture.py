"""Verify labeled regional Anima output workflows and result persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from tools.anima_regional_output_capture.matrix import (
    CFG,
    HEIGHT,
    SEED,
    STEPS,
    WIDTH,
    profiles,
)
from tools.anima_regional_output_capture.results import VisualOutputResultRecorder
from tools.anima_regional_output_capture.workflow import VisualOutputWorkflowBuilder
from tools.comfy_api import ImageReference, JsonObject


def test_visual_matrix_builds_global_fidelity_and_true_split_region_graphs() -> None:
    """Keep fixed labels, settings, adapter ownership, and masks in each graph."""

    definitions = profiles()
    builder = VisualOutputWorkflowBuilder()
    global_reference = builder.build(definitions[0], run_id="visual-run", mask_names=())
    all_one = builder.build(definitions[1], run_id="visual-run", mask_names=())
    split = builder.build(
        definitions[3],
        run_id="visual-run",
        mask_names=("left.png", "right.png"),
    )

    reference_types = _types(global_reference.prompt)
    assert "LoraLoaderModelOnly" in reference_types
    assert "SimpleSyrupBenchmark.StaticAnimaRegionalProfile" not in reference_types
    all_one_probe = _sole(
        all_one.prompt, "SimpleSyrupBenchmark.StaticAnimaRegionalProfile"
    )
    all_one_assignments = json.loads(
        cast(
            str,
            cast(dict[str, object], all_one_probe["inputs"])["regional_adapters_json"],
        )
    )
    assert all_one_assignments == [
        {
            "region_index": 0,
            "branch": "positive",
            "lora_name": "Anima\\style\\adapter-a.safetensors",
            "strength": 1.0,
        }
    ]
    split_probe = _sole(split.prompt, "SimpleSyrupBenchmark.StaticAnimaRegionalProfile")
    split_assignments = json.loads(
        cast(
            str,
            cast(dict[str, object], split_probe["inputs"])["regional_adapters_json"],
        )
    )
    assert [
        (item["region_index"], item["lora_name"]) for item in split_assignments
    ] == [
        (0, "Anima\\style\\adapter-a.safetensors"),
        (1, "Anima\\style\\adapter-b.safetensors"),
    ]
    mask_loader = _sole(split.prompt, "SimpleSyrup.LoadMaskBatch")
    assert cast(dict[str, object], mask_loader["inputs"])["image"] == {
        "__value__": ["left.png", "right.png"]
    }
    sampler = _sole(split.prompt, "KSampler")
    sampler_inputs = cast(dict[str, object], sampler["inputs"])
    assert (sampler_inputs["seed"], sampler_inputs["steps"], sampler_inputs["cfg"]) == (
        SEED,
        STEPS,
        CFG,
    )
    assert (WIDTH, HEIGHT) == (1024, 1024)


def test_visual_result_recorder_persists_exact_labels_and_hashes(
    tmp_path: Path,
) -> None:
    """Associate each image with its immutable profile and complete sidecars."""

    definitions = profiles()
    recorder = VisualOutputResultRecorder(tmp_path)
    workflow: dict[str, JsonObject] = {"1": {"class_type": "SaveImage", "inputs": {}}}
    history: JsonObject = {"status": {"completed": True}}
    for index, profile in enumerate(definitions):
        recorder.record(
            profile,
            workflow=workflow,
            history=history,
            prompt_id=f"prompt-{index}",
            image_reference=ImageReference(f"server-{index}.png", "visual", "output"),
            image_bytes=f"png-{index}".encode(),
        )
    result_path = recorder.finalize(definitions)
    result = cast(
        JsonObject,
        json.loads(result_path.read_text(encoding="utf-8")),
    )

    assert result["status"] == "completed"
    observations = cast(list[JsonObject], result["observations"])
    assert [item["profile_id"] for item in observations] == [
        profile.profile_id for profile in definitions
    ]
    split = observations[-1]
    assert split["label"] == "Global GLOBAL_ADAPTER with left ADAPTER_A and right ADAPTER_B"
    assert (tmp_path / cast(str, split["image_file"])).read_bytes() == b"png-3"
    assert len(cast(str, split["image_sha256"])) == 64

    with pytest.raises(ValueError, match="already recorded"):
        recorder.record(
            definitions[-1],
            workflow=workflow,
            history=history,
            prompt_id="duplicate",
            image_reference=ImageReference("duplicate.png", "visual", "output"),
            image_bytes=b"duplicate",
        )


def _types(prompt: dict[str, JsonObject]) -> set[str]:
    """Return every graph node class name."""

    return {str(node["class_type"]) for node in prompt.values()}


def _sole(prompt: dict[str, JsonObject], class_type: str) -> JsonObject:
    """Return the sole graph node of one required class."""

    matches = [node for node in prompt.values() if node["class_type"] == class_type]
    assert len(matches) == 1
    return matches[0]
