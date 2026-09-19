# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run isolated live sampling proof for automatic NegPiP on every family."""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from typing import cast

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.default_paths import default_comfy_root
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_model_links import (
    ManagedComfyModelLinks,
    ManagedModelLink,
)
from tools.comfy_integration.managed_server import ManagedComfyServer

from .upstream_parity import prove_upstream_parity
from .visual_proof import NegpipVisualProofRecorder
from .workflow import (
    BuiltNegpipLiveWorkflow,
    NegpipFixtureSelections,
    NegpipLiveFamily,
    NegpipLiveWorkflowBuilder,
)

MODEL_LIBRARY_ENVIRONMENT_VARIABLE = "SIMPLE_SYRUP_MODEL_LIBRARY"
REFINER_FIXTURE_ENVIRONMENT_VARIABLE = "SIMPLE_SYRUP_SDXL_REFINER_FIXTURE"
COMFY_ROOT = default_comfy_root()
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PPM_ROOT = REPOSITORY_ROOT / ".codex" / "references" / "ComfyUI-ppm-6c6c3601"
ARTIFACT_ROOT = COMFY_ROOT / "benchmark_artifacts" / "negpip-automatic"
SELECTIONS = NegpipFixtureSelections(
    sd1_checkpoint=r"simple_syrup_negpip\sd1.safetensors",
    sdxl_checkpoint=r"simple_syrup_negpip\sdxl.safetensors",
    sdxl_refiner_checkpoint=r"simple_syrup_negpip\sdxl-refiner.safetensors",
    anima_diffusion=r"simple_syrup_negpip\anima.safetensors",
    anima_text_encoder=r"simple_syrup_negpip\anima-text-encoder.safetensors",
    krea2_diffusion=r"simple_syrup_negpip\krea2.safetensors",
    krea2_text_encoder=r"simple_syrup_negpip\krea2-text-encoder.safetensors",
    qwen_image_vae=r"simple_syrup_negpip\qwen-image-vae.safetensors",
)
PPM_BASELINE_FAMILIES = frozenset(
    {
        NegpipLiveFamily.SD1,
        NegpipLiveFamily.SDXL,
        NegpipLiveFamily.SDXL_REFINER,
        NegpipLiveFamily.ANIMA,
    }
)


def execute() -> Path:
    """Run controls and one sampled negative-weight case per model family."""

    artifacts = IntegrationArtifacts(ARTIFACT_ROOT)
    model_library = _model_library_root()
    refiner_fixture = _refiner_fixture_path()
    builder = NegpipLiveWorkflowBuilder(SELECTIONS)
    visual_recorder = NegpipVisualProofRecorder(artifacts.root)
    workflows = tuple(
        workflow
        for family in NegpipLiveFamily
        for workflow in _family_workflows(builder, artifacts.run_id, family)
    )
    required = frozenset().union(
        *(workflow.required_node_ids for workflow in workflows)
    )
    links = ManagedComfyModelLinks(
        model_root=COMFY_ROOT / "models",
        links=_model_links(model_library, refiner_fixture),
    )
    parity = prove_upstream_parity(PPM_ROOT)
    result: dict[str, object] = {
        "run_id": artifacts.run_id,
        "upstream_parity": parity,
        "families": {},
    }
    port: int | None = None
    try:
        with ExitStack() as stack:
            stack.enter_context(links)
            running = stack.enter_context(
                ManagedComfyServer(
                    comfy_root=COMFY_ROOT,
                    artifacts=artifacts,
                    required_node_ids=required,
                    readiness_timeout=300.0,
                    launch_arguments=(
                        "--disable-all-custom-nodes",
                        "--whitelist-custom-nodes",
                        "SimpleSyrup",
                        "SimpleSyrupBenchmarkProbe",
                        "comfyui-prompt-control",
                        "comfyui-ppm",
                    ),
                )
            )
            port = running.port
            result["port"] = port
            result["isolated_custom_nodes"] = [
                "SimpleSyrup",
                "SimpleSyrupBenchmarkProbe",
                "comfyui-prompt-control",
                "comfyui-ppm",
            ]
            family_results = cast(dict[str, object], result["families"])
            for workflow in workflows:
                prompt_id = running.client.submit(workflow.prompt)
                history = running.client.wait_for_history(prompt_id, timeout=1800.0)
                evidence = _parse_history(history, workflow)
                evidence["prompt_id"] = prompt_id
                mode = workflow.mode
                reference = extract_saved_image(history, workflow.image_node_id)
                evidence["image"] = visual_recorder.record(
                    workflow.family,
                    mode,
                    running.client.download_image(reference),
                )
                family = cast(
                    dict[str, object],
                    family_results.setdefault(
                        workflow.family.value,
                        {},
                    ),
                )
                family[mode] = evidence
                (
                    artifacts.root / f"{workflow.family.value}-{mode}.history.json"
                ).write_text(
                    json.dumps(history, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        if not links.cleaned:
            raise RuntimeError("Managed model aliases were not cleaned.")
        if port is None or not is_loopback_port_available(port):
            raise RuntimeError("Managed Comfy custom port was not released.")
        result["visual_proof"] = visual_recorder.finalize(
            cast(JsonObject, result["families"])
        )
        _validate_complete_result(result)
        proof_path = artifacts.root / "negpip-proof.json"
        proof_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts.record_cleanup(process_running=False, port_available=True)
        return proof_path
    except BaseException as error:
        artifacts.record_failure(error)
        raise


def _model_links(
    model_library: Path,
    refiner_fixture: Path,
) -> tuple[ManagedModelLink, ...]:
    """Return exact external fixtures and temporary Comfy selection aliases."""

    return (
        ManagedModelLink(
            model_library
            / "checkpoints"
            / "SD 1.5"
            / "abyssorangemix3AOM3_aom3a3.safetensors",
            "checkpoints",
            SELECTIONS.sd1_checkpoint,
        ),
        ManagedModelLink(
            model_library
            / "checkpoints"
            / "SDXL"
            / "juggernautXL_juggXIByRundiffusion.safetensors",
            "checkpoints",
            SELECTIONS.sdxl_checkpoint,
        ),
        ManagedModelLink(
            refiner_fixture,
            "checkpoints",
            SELECTIONS.sdxl_refiner_checkpoint,
        ),
        ManagedModelLink(
            model_library / "diffusion_models" / "Anima" / "anima_baseV10.safetensors",
            "diffusion_models",
            SELECTIONS.anima_diffusion,
        ),
        ManagedModelLink(
            model_library / "text_encoders" / "qwen" / "qwen_3_06b_base.safetensors",
            "text_encoders",
            SELECTIONS.anima_text_encoder,
        ),
        ManagedModelLink(
            model_library
            / "diffusion_models"
            / "Krea2"
            / "redcraftHybridH3Krea2dual_11INT8INT4_fp8.safetensors",
            "diffusion_models",
            SELECTIONS.krea2_diffusion,
        ),
        ManagedModelLink(
            model_library
            / "text_encoders"
            / "qwen"
            / "qwen3vl_4b_fp8_scaled.safetensors",
            "text_encoders",
            SELECTIONS.krea2_text_encoder,
        ),
        ManagedModelLink(
            model_library / "VAE" / "qwen" / "qwen_image_vae.safetensors",
            "vae",
            SELECTIONS.qwen_image_vae,
        ),
    )


def _family_workflows(
    builder: NegpipLiveWorkflowBuilder,
    run_id: str,
    family: NegpipLiveFamily,
) -> tuple[BuiltNegpipLiveWorkflow, ...]:
    """Keep each control, automatic, and PPM oracle execution adjacent."""

    workflows = [
        builder.build(
            family,
            run_id=f"{run_id}:{family.value}:control",
            trigger=False,
        ),
        builder.build(
            family,
            run_id=f"{run_id}:{family.value}:negative",
            trigger=True,
        ),
    ]
    if family in PPM_BASELINE_FAMILIES:
        workflows.append(
            builder.build(
                family,
                run_id=f"{run_id}:{family.value}:ppm-baseline",
                trigger=True,
                baseline_ppm=True,
            )
        )
    return tuple(workflows)


def _model_library_root() -> Path:
    """Return the explicit absolute external fixture library root."""

    configured = os.environ.get(MODEL_LIBRARY_ENVIRONMENT_VARIABLE)
    if not configured:
        raise RuntimeError(
            f"Set {MODEL_LIBRARY_ENVIRONMENT_VARIABLE} to the model fixture root."
        )
    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise ValueError(
            f"{MODEL_LIBRARY_ENVIRONMENT_VARIABLE} must be an absolute path."
        )
    if not root.is_dir():
        raise FileNotFoundError("Configured model fixture library does not exist.")
    return root.resolve()


def _refiner_fixture_path() -> Path:
    """Return the explicit SDXL Refiner checkpoint used by live proof."""

    configured = os.environ.get(REFINER_FIXTURE_ENVIRONMENT_VARIABLE)
    if not configured:
        raise RuntimeError(
            f"Set {REFINER_FIXTURE_ENVIRONMENT_VARIABLE} to a refiner checkpoint."
        )
    fixture = Path(configured).expanduser()
    if not fixture.is_absolute():
        raise ValueError(
            f"{REFINER_FIXTURE_ENVIRONMENT_VARIABLE} must be an absolute path."
        )
    if not fixture.is_file():
        raise FileNotFoundError("Configured SDXL Refiner fixture does not exist.")
    return fixture.resolve()


def _parse_history(
    history: JsonObject,
    workflow: BuiltNegpipLiveWorkflow,
) -> JsonObject:
    """Extract exact control or sampled runtime evidence from completed history."""

    status = _mapping(history.get("status"), "history.status")
    if status.get("status_str") != "success" or status.get("completed") is not True:
        raise RuntimeError(
            f"Managed NegPiP workflow failed: {status.get('messages')!r}"
        )
    outputs = _mapping(history.get("outputs"), "history.outputs")
    modifier = _single_output(
        outputs,
        workflow.modifier_node_id,
        "model_modifier_snapshot",
    )
    result: JsonObject = {"modifier": modifier}
    if workflow.runtime_node_id is None:
        return result
    if workflow.conditioning_node_id is None:
        raise ValueError("Triggered NegPiP workflow is missing conditioning evidence.")
    result["runtime"] = _single_output(
        outputs,
        workflow.runtime_node_id,
        "negpip_runtime_evidence",
    )
    result["conditioning"] = _single_output(
        outputs,
        workflow.conditioning_node_id,
        "conditioning_batch_snapshot",
    )
    return result


def _single_output(
    outputs: JsonObject,
    node_id: str,
    field: str,
) -> JsonObject:
    """Return one exact UI evidence record."""

    node = _mapping(outputs.get(node_id), f"outputs[{node_id}]")
    values = node.get(field)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"{field} must contain exactly one record.")
    return _mapping(values[0], field)


def _mapping(value: object, field: str) -> JsonObject:
    """Narrow one JSON object with string keys."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a JSON object.")
    return cast(JsonObject, value)


def _validate_complete_result(result: dict[str, object]) -> None:
    """Fail unless gating and live negative execution passed for every family."""

    families = cast(dict[str, object], result["families"])
    if set(families) != {family.value for family in NegpipLiveFamily}:
        raise ValueError("Managed NegPiP proof did not cover every model family.")
    for family_name, family_value in families.items():
        family = cast(dict[str, object], family_value)
        control = cast(dict[str, object], family["control"])
        negative = cast(dict[str, object], family["negative"])
        control_modifier = cast(dict[str, object], control["modifier"])
        negative_modifier = cast(dict[str, object], negative["modifier"])
        runtime = cast(dict[str, object], negative["runtime"])
        control_image = cast(dict[str, object], control["image"])
        negative_image = cast(dict[str, object], negative["image"])
        comparison = cast(dict[str, object], family["image_comparison"])
        if control_modifier.get("ppm_negpip") is not False:
            raise ValueError(f"{family_name} control unexpectedly enabled NegPiP.")
        if negative_modifier.get("ppm_negpip") is not True:
            raise ValueError(f"{family_name} negative prompt did not enable NegPiP.")
        if (
            not isinstance(runtime.get("attention_calls"), int)
            or cast(int, runtime["attention_calls"]) < 1
        ):
            raise ValueError(f"{family_name} did not execute NegPiP attention.")
        if (
            not isinstance(runtime.get("negative_mask_calls"), int)
            or cast(int, runtime["negative_mask_calls"]) < 1
        ):
            raise ValueError(f"{family_name} never applied a negative token sign.")
        if runtime.get("invariant_failures") != []:
            raise ValueError(f"{family_name} reported live NegPiP invariant failures.")
        if control_image.get("rgb_sha256") == negative_image.get("rgb_sha256"):
            raise ValueError(f"{family_name} decoded image pair is identical.")
        changed_pixels = comparison.get("changed_pixels")
        if not isinstance(changed_pixels, int) or changed_pixels < 1:
            raise ValueError(f"{family_name} has no visible pixel differences.")
        if family_name in {item.value for item in PPM_BASELINE_FAMILIES}:
            baseline = cast(dict[str, object], family["ppm_baseline"])
            baseline_conditioning = cast(dict[str, object], baseline["conditioning"])
            negative_conditioning = cast(dict[str, object], negative["conditioning"])
            parity = cast(dict[str, object], family["ppm_automatic_comparison"])
            mean_delta = parity.get("mean_absolute_rgb_delta")
            p99_delta = parity.get("p99_channel_delta")
            if (
                not isinstance(mean_delta, (int, float))
                or mean_delta > 0.5
                or not isinstance(p99_delta, (int, float))
                or p99_delta > 5.0
            ):
                raise ValueError(
                    f"{family_name} automatic image exceeded pinned PPM numerical "
                    "parity tolerance."
                )
            for side in ("positive", "negative"):
                if baseline_conditioning.get(side) != negative_conditioning.get(side):
                    raise ValueError(
                        f"{family_name} {side} conditioning diverged from pinned PPM."
                    )
            baseline_runtime = cast(dict[str, object], baseline["runtime"])
            for field in (
                "family",
                "patch_name",
                "attention_calls",
                "negative_mask_calls",
                "input_value_shape",
                "output_value_shape",
                "mask_shape",
                "text_length",
                "negative_token_count",
                "negative_token_positions",
                "negative_token_locations",
                "invariant_failures",
            ):
                if baseline_runtime.get(field) != runtime.get(field):
                    raise ValueError(
                        f"{family_name} live {field} diverged from pinned PPM."
                    )


if __name__ == "__main__":
    print(execute())
