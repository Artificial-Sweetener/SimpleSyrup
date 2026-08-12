"""Decode benchmark manifest JSON into strongly typed records."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypeVar, cast

from .manifest_types import (
    BenchmarkCase,
    BenchmarkManifest,
    ExecutionProfile,
    JsonObject,
    MaskRectangle,
    ModelArtifact,
    SamplingSettings,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RecordT = TypeVar("RecordT")


def decode_manifest(payload: object) -> BenchmarkManifest:
    """Decode one manifest payload while rejecting undeclared JSON fields."""

    root = json_object(payload, "manifest")
    exact_keys(
        root,
        {
            "schema_version",
            "benchmark_id",
            "source",
            "models",
            "sampling",
            "executions",
            "cases",
            "artifact_name_template",
        },
        "manifest",
    )
    source = json_object(root["source"], "source")
    exact_keys(source, {"simple_syrup_commit", "comfyui_commit"}, "source")
    return BenchmarkManifest(
        schema_version=integer(root["schema_version"], "schema_version", minimum=1),
        benchmark_id=identifier(root["benchmark_id"], "benchmark_id"),
        simple_syrup_commit=commit(source["simple_syrup_commit"], "SimpleSyrup"),
        comfyui_commit=commit(source["comfyui_commit"], "ComfyUI"),
        models=_records(root["models"], "models", _model),
        sampling=_sampling(root["sampling"]),
        executions=_records(root["executions"], "executions", _execution),
        cases=_records(root["cases"], "cases", _case),
        artifact_name_template=text(
            root["artifact_name_template"], "artifact_name_template"
        ),
    )


def _model(value: object, label: str) -> ModelArtifact:
    """Decode one stable model-stack identity."""

    entry = json_object(value, label)
    exact_keys(entry, {"id", "role", "filename", "size_bytes", "sha256"}, label)
    checksum = text(entry["sha256"], f"{label}.sha256")
    if SHA256_PATTERN.fullmatch(checksum) is None:
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256.")
    return ModelArtifact(
        artifact_id=identifier(entry["id"], f"{label}.id"),
        role=identifier(entry["role"], f"{label}.role"),
        filename=text(entry["filename"], f"{label}.filename"),
        size_bytes=integer(entry["size_bytes"], f"{label}.size_bytes", minimum=1),
        sha256=checksum,
    )


def _sampling(value: object) -> SamplingSettings:
    """Decode the shared fixed sampling configuration."""

    entry = json_object(value, "sampling")
    exact_keys(
        entry,
        {
            "width",
            "height",
            "steps",
            "cfg",
            "sampler",
            "scheduler",
            "denoise",
            "seeds",
            "negative_prompt",
        },
        "sampling",
    )
    return SamplingSettings(
        width=integer(entry["width"], "sampling.width", minimum=8),
        height=integer(entry["height"], "sampling.height", minimum=8),
        steps=integer(entry["steps"], "sampling.steps", minimum=1),
        cfg=number(entry["cfg"], "sampling.cfg", minimum=0.0),
        sampler=identifier(entry["sampler"], "sampling.sampler"),
        scheduler=identifier(entry["scheduler"], "sampling.scheduler"),
        denoise=number(entry["denoise"], "sampling.denoise", minimum=0.0),
        seeds=tuple(
            integer(seed, f"sampling.seeds[{index}]", minimum=0)
            for index, seed in enumerate(json_array(entry["seeds"], "sampling.seeds"))
        ),
        negative_prompt=text(entry["negative_prompt"], "sampling.negative_prompt"),
    )


def _execution(value: object, label: str) -> ExecutionProfile:
    """Decode one spatial execution profile."""

    entry = json_object(value, label)
    exact_keys(entry, {"id", "strategy", "spatial_mode", "controls"}, label)
    return ExecutionProfile(
        execution_id=identifier(entry["id"], f"{label}.id"),
        strategy=identifier(entry["strategy"], f"{label}.strategy"),
        spatial_mode=identifier(entry["spatial_mode"], f"{label}.spatial_mode"),
        controls=json_object(entry["controls"], f"{label}.controls"),
    )


def _case(value: object, label: str) -> BenchmarkCase:
    """Decode one prompt and normalized-mask scenario."""

    entry = json_object(value, label)
    exact_keys(
        entry,
        {
            "id",
            "scenario_tags",
            "global_prompt",
            "regional_prompts",
            "masks",
            "regional_prompt_weight",
            "region_mask_feather",
        },
        label,
    )
    prompts = tuple(
        text(prompt, f"{label}.regional_prompts[{index}]")
        for index, prompt in enumerate(
            json_array(entry["regional_prompts"], f"{label}.regional_prompts")
        )
    )
    masks = _records(entry["masks"], f"{label}.masks", _mask)
    if len(prompts) != len(masks):
        raise ValueError(f"{label} must pair every regional prompt with one mask.")
    return BenchmarkCase(
        case_id=identifier(entry["id"], f"{label}.id"),
        scenario_tags=tuple(
            identifier(tag, f"{label}.scenario_tags[{index}]")
            for index, tag in enumerate(
                json_array(entry["scenario_tags"], f"{label}.scenario_tags")
            )
        ),
        global_prompt=text(entry["global_prompt"], f"{label}.global_prompt"),
        regional_prompts=prompts,
        masks=masks,
        regional_prompt_weight=number(
            entry["regional_prompt_weight"],
            f"{label}.regional_prompt_weight",
            minimum=0.0,
        ),
        region_mask_feather=integer(
            entry["region_mask_feather"],
            f"{label}.region_mask_feather",
            minimum=0,
        ),
    )


def _mask(value: object, label: str) -> MaskRectangle:
    """Decode one positive-area normalized mask rectangle."""

    entry = json_object(value, label)
    exact_keys(entry, {"x0", "y0", "x1", "y1"}, label)
    rectangle = MaskRectangle(
        x0=number(entry["x0"], f"{label}.x0", minimum=0.0),
        y0=number(entry["y0"], f"{label}.y0", minimum=0.0),
        x1=number(entry["x1"], f"{label}.x1", minimum=0.0),
        y1=number(entry["y1"], f"{label}.y1", minimum=0.0),
    )
    if rectangle.x1 > 1.0 or rectangle.y1 > 1.0:
        raise ValueError(f"{label} coordinates must not exceed 1.0.")
    if rectangle.x0 >= rectangle.x1 or rectangle.y0 >= rectangle.y1:
        raise ValueError(f"{label} must have positive normalized area.")
    return rectangle


def _records(
    value: object,
    label: str,
    decoder: Callable[[object, str], RecordT],
) -> tuple[RecordT, ...]:
    """Decode one array of labeled records."""

    return tuple(
        decoder(item, f"{label}[{index}]")
        for index, item in enumerate(json_array(value, label))
    )


def json_object(value: object, label: str) -> JsonObject:
    """Narrow one JSON object boundary."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys.")
    return cast(JsonObject, value)


def json_array(value: object, label: str) -> list[object]:
    """Narrow one JSON array boundary."""

    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array.")
    return cast(list[object], value)


def text(value: object, label: str) -> str:
    """Narrow one required nonempty string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value


def identifier(value: object, label: str) -> str:
    """Narrow one lowercase filename-safe identifier."""

    result = text(value, label)
    if re.fullmatch(r"[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?", result) is None:
        raise ValueError(f"{label} must be a lowercase filename-safe identifier.")
    return result


def integer(value: object, label: str, *, minimum: int) -> int:
    """Narrow one bounded integer while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer of at least {minimum}.")
    return value


def number(value: object, label: str, *, minimum: float) -> float:
    """Narrow one finite bounded number while rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if result < minimum or result in {float("inf"), float("-inf")} or result != result:
        raise ValueError(f"{label} must be finite and at least {minimum}.")
    return result


def commit(value: object, repository: str) -> str:
    """Validate one full lowercase Git commit identity."""

    result = text(value, f"{repository} commit")
    if re.fullmatch(r"[0-9a-f]{40}", result) is None:
        raise ValueError(f"{repository} commit must be a full lowercase SHA-1.")
    return result


def exact_keys(value: JsonObject, expected: set[str], label: str) -> None:
    """Reject missing and undeclared manifest fields."""

    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
