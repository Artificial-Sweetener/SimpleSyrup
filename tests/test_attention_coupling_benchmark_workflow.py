"""Verify API-format workflow construction for every benchmark spatial mode."""

import pytest

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.workflow import BenchmarkWorkflowBuilder


@pytest.mark.parametrize(
    ("execution_id", "class_type", "diffusion_mode"),
    [
        (
            "regional-conditioning-full",
            "SimpleSyrup.KSamplerPromptByRegion",
            None,
        ),
        (
            "regional-conditioning-multidiffusion",
            "SimpleSyrup.KSamplerPromptByTiledRegion",
            "multidiffusion",
        ),
        (
            "regional-conditioning-mod",
            "SimpleSyrup.KSamplerPromptByTiledRegion",
            "mixture_of_diffusers",
        ),
        (
            "regional-conditioning-contextual-md",
            "SimpleSyrup.KSamplerContextualDiffusion",
            "multidiffusion",
        ),
        (
            "regional-conditioning-contextual-mod",
            "SimpleSyrup.KSamplerContextualDiffusion",
            "mixture_of_diffusers",
        ),
    ],
)
def test_workflow_uses_public_sampler_contract_for_each_execution(
    execution_id: str,
    class_type: str,
    diffusion_mode: str | None,
) -> None:
    """Build each spatial profile through its current public node schema."""

    manifest = load_manifest()
    run = next(
        run
        for run in manifest.runs()
        if run.case_id == "vertical-hard-50-50" and run.execution_id == execution_id
    )
    built = BenchmarkWorkflowBuilder().build(
        manifest,
        run,
        ("benchmark/left.png", "benchmark/right.png"),
    )
    sampler = built.prompt["6"]
    inputs = sampler["inputs"]

    assert sampler["class_type"] == class_type
    assert isinstance(inputs, dict)
    assert inputs["seed"] == run.seed
    assert inputs["sampler_name"] == "er_sde"
    assert inputs["scheduler"] == "simple"
    assert inputs["region_masks"] == ["3", 0]
    assert inputs.get("diffusion_mode") == diffusion_mode
    assert built.prompt["5"]["class_type"] == "SimpleSyrupBenchmark.InstrumentModel"
    assert built.prompt["7"]["class_type"] == "SimpleSyrupBenchmark.ReadMetrics"
    assert built.prompt["9"]["class_type"] == "SaveImage"


def test_workflow_serializes_global_first_prompts_and_ordered_masks() -> None:
    """Keep prompt and mask order aligned without modifying manifest text."""

    manifest = load_manifest()
    run = next(
        run for run in manifest.runs() if run.case_id == "short-regional-prompts"
    )
    built = BenchmarkWorkflowBuilder().build(
        manifest,
        run,
        ("benchmark/left.png", "benchmark/right.png"),
    )
    encode_inputs = built.prompt["2"]["inputs"]
    mask_inputs = built.prompt["3"]["inputs"]

    assert isinstance(encode_inputs, dict)
    assert encode_inputs["positive_prompt"] == (
        "masterpiece, best quality, score_7, safe, two portraits"
        "[SEP]red-haired girl[SEP]blue-haired boy"
    )
    assert isinstance(mask_inputs, dict)
    assert mask_inputs == {
        "image": {
            "__value__": ["benchmark/left.png", "benchmark/right.png"],
        },
        "channel": "red",
    }


def test_global_only_workflow_uses_zero_technical_mask() -> None:
    """Run the no-region control without inventing an authored regional input."""

    manifest = load_manifest()
    run = next(run for run in manifest.runs() if run.case_id == "global-only-control")

    built = BenchmarkWorkflowBuilder().build(manifest, run, ())

    assert built.prompt["3"] == {
        "class_type": "SolidMask",
        "inputs": {"value": 0.0, "width": 1024, "height": 1024},
    }
