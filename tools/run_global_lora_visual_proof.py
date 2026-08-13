# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the fixed-seed global and regional ADAPTER_A visual comparison."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw

from tools.comfy_api import ImageReference, JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.global_lora_visual_proof.matrix import (
    ADAPTER_A_NAME,
    GlobalLoraVisualCase,
    cases,
    render_positive_prompt,
)
from tools.global_lora_visual_proof.results import GlobalLoraVisualProofRecorder

LOGGER = logging.getLogger(__name__)
COMFY_ROOT = Path(r"<COMFY_ROOT>")
OUTPUT_ROOT = (
    COMFY_ROOT
    / "benchmark_artifacts"
    / "anima-regional-prompting-v1"
    / "global-lora-visual-proof"
)
SOURCE_WORKFLOW = (
    COMFY_ROOT
    / "benchmark_artifacts"
    / "anima-regional-prompting-v1"
    / "user-prompt-checkpoint_a-regional-lora-pair"
    / "20260812T160702Z-ad444583"
    / "checkpoint_a-adapter_a-pink-only__workflow.json"
)


def main() -> int:
    """Execute all cases in one managed server and preserve labeled evidence."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(OUTPUT_ROOT)
    mask_paths: tuple[Path, ...] = ()
    try:
        source = _source_prompt()
        definitions = cases()
        mask_paths = _write_masks(artifacts.run_id)
        workflows = tuple(
            _workflow(source, case, artifacts.run_id, mask_paths)
            for case in definitions
        )
        required = frozenset(
            cast(str, node["class_type"])
            for workflow in workflows
            for node in workflow.values()
        )
        recorder = GlobalLoraVisualProofRecorder(artifacts.root)
        first_success: (
            tuple[dict[str, JsonObject], JsonObject, str, ImageReference, bytes] | None
        ) = None
        with ManagedComfyServer(
            comfy_root=COMFY_ROOT,
            artifacts=artifacts,
            required_node_ids=required,
            readiness_timeout=240.0,
        ) as running:
            for case, workflow in zip(definitions, workflows, strict=True):
                LOGGER.info("Starting %s", case.label)
                prompt_id = running.client.submit(workflow)
                history = running.client.wait_for_history(prompt_id, timeout=1200.0)
                if case.expect_overlap_rejection:
                    recorder.record_overlap_rejection(
                        case,
                        workflow=workflow,
                        history=history,
                        prompt_id=prompt_id,
                    )
                    continue
                reference = extract_saved_image(history, "11")
                image_bytes = running.client.download_image(reference)
                recorder.record_success(
                    case,
                    workflow=workflow,
                    history=history,
                    prompt_id=prompt_id,
                    reference=reference,
                    image_bytes=image_bytes,
                )
                if first_success is None:
                    first_success = (
                        workflow,
                        history,
                        prompt_id,
                        reference,
                        image_bytes,
                    )
            system_stats = running.system_stats
            port = running.port
            process = running.process
        if first_success is None:
            raise RuntimeError("Global LoRA proof produced no successful image.")
        workflow, history, prompt_id, reference, image_bytes = first_success
        artifacts.record_success(
            system_stats=system_stats,
            workflow=workflow,
            history=history,
            prompt_id=prompt_id,
            image_reference=reference,
            image_bytes=image_bytes,
        )
        for path in mask_paths:
            path.unlink(missing_ok=True)
        masks_removed = not any(path.exists() for path in mask_paths)
        port_available = is_loopback_port_available(port)
        artifacts.record_cleanup(
            process_running=process.is_running,
            port_available=port_available,
        )
        result = recorder.finalize(
            definitions,
            system_stats=system_stats,
            cleanup_verified=not process.is_running and port_available,
            masks_removed=masks_removed,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Global LoRA visual proof failed at %s", artifacts.root)
        return 1
    finally:
        for path in mask_paths:
            path.unlink(missing_ok=True)
    LOGGER.info("Global LoRA visual proof completed at %s", result)
    return 0


def _source_prompt() -> dict[str, JsonObject]:
    """Load the exact fixed-seed user-prompt source workflow."""

    decoded = json.loads(SOURCE_WORKFLOW.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError("Global LoRA proof source workflow must be an object.")
    return cast(dict[str, JsonObject], decoded)


def _workflow(
    source: dict[str, JsonObject],
    case: GlobalLoraVisualCase,
    run_id: str,
    masks: tuple[Path, ...],
) -> dict[str, JsonObject]:
    """Return one isolated run-local graph for a placement case."""

    prompt = copy.deepcopy(source)
    inputs = prompt["2"]["inputs"]
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("positive_prompt"), str
    ):
        raise TypeError("Global LoRA proof source positive prompt is unavailable.")
    inputs["positive_prompt"] = render_positive_prompt(
        cast(str, inputs["positive_prompt"]), case
    )
    if case.global_strength is not None:
        prompt["12"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": ADAPTER_A_NAME,
                "strength_model": case.global_strength,
            },
        }
        inputs["model"] = ["12", 0]
    else:
        inputs["model"] = ["1", 0]
    prompt["3"]["inputs"] = {
        "channel": "red",
        "image": {"__value__": [path.name for path in masks]},
    }
    for node, suffix, link in (
        ("5", "metrics", ["2", 0]),
        ("6", "regional-diagnostics", ["5", 0]),
    ):
        prompt[node]["inputs"] = {
            "model": link,
            "run_id": f"{run_id}:{case.case_id}:{suffix}",
        }
    prompt["8"]["inputs"] = {
        "latent": ["7", 0],
        "run_id": f"{run_id}:{case.case_id}:metrics",
    }
    prompt["9"]["inputs"] = {
        "latent": ["8", 0],
        "run_id": f"{run_id}:{case.case_id}:regional-diagnostics",
    }
    prompt["11"]["inputs"] = {
        "filename_prefix": f"simple_syrup_global_lora/{run_id}/{case.case_id}",
        "images": ["10", 0],
    }
    return prompt


def _write_masks(run_id: str) -> tuple[Path, Path]:
    """Write exact hard left/right masks under unique run-local names."""

    paths = tuple(
        COMFY_ROOT / "input" / f"{run_id.lower()}__global-lora-region-{index:02d}.png"
        for index in range(2)
    )
    for index, path in enumerate(paths):
        image = Image.new("RGB", (1024, 1024), "black")
        ImageDraw.Draw(image).rectangle(
            (0 if index == 0 else 512, 0, 511 if index == 0 else 1023, 1023),
            fill="white",
        )
        image.save(path, format="PNG", compress_level=9)
    return cast(tuple[Path, Path], paths)


if __name__ == "__main__":
    raise SystemExit(main())
