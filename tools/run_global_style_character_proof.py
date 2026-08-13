# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run global ADAPTER_A style with a right-region CHARACTER_A character LoRA."""

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
from tools.global_style_character_proof.matrix import (
    ADAPTER_A_NAME,
    GlobalStyleCharacterCase,
    cases,
)
from tools.global_style_character_proof.results import (
    GlobalStyleCharacterProofRecorder,
)

LOGGER = logging.getLogger(__name__)
COMFY_ROOT = Path(r"<COMFY_ROOT>")
OUTPUT_ROOT = (
    COMFY_ROOT
    / "benchmark_artifacts"
    / "anima-regional-prompting-v1"
    / "global-style-character-proof"
)
SOURCE_RUN = (
    COMFY_ROOT
    / "benchmark_artifacts"
    / "anima-regional-prompting-v1"
    / "user-prompt-character_a-ownership-proof"
    / "20260812T191525Z-6a2f4a67"
    / "run.json"
)


def main() -> int:
    """Execute three fixed-seed cases in one managed Comfy instance."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(OUTPUT_ROOT)
    mask_paths: tuple[Path, ...] = ()
    try:
        source = _source_workflow()
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
        recorder = GlobalStyleCharacterProofRecorder(artifacts.root)
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
                reference = extract_saved_image(history, "11")
                image_bytes = running.client.download_image(reference)
                recorder.record(
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
            raise RuntimeError("Global style/character proof produced no image.")
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
        LOGGER.exception("Global style/character proof failed at %s", artifacts.root)
        return 1
    finally:
        for path in mask_paths:
            path.unlink(missing_ok=True)
    LOGGER.info("Global style/character proof completed at %s", result)
    return 0


def _source_workflow() -> dict[str, JsonObject]:
    """Load the exact selected CHARACTER_A proof workflow."""

    decoded = json.loads(SOURCE_RUN.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or not isinstance(decoded.get("workflow"), dict):
        raise TypeError("Selected CHARACTER_A run lacks its workflow object.")
    return cast(dict[str, JsonObject], decoded["workflow"])


def _workflow(
    source: dict[str, JsonObject],
    case: GlobalStyleCharacterCase,
    run_id: str,
    masks: tuple[Path, ...],
) -> dict[str, JsonObject]:
    """Add optional whole-image ADAPTER_A without changing regional CHARACTER_A hooks."""

    prompt = copy.deepcopy(source)
    encoder_inputs = prompt["2"]["inputs"]
    if not isinstance(encoder_inputs, dict):
        raise TypeError("Selected CHARACTER_A encoder inputs must be an object.")
    if case.global_style_strength is not None:
        prompt["12"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": ADAPTER_A_NAME,
                "strength_model": case.global_style_strength,
            },
        }
        encoder_inputs["model"] = ["12", 0]
    else:
        encoder_inputs["model"] = ["1", 0]
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
        "filename_prefix": f"simple_syrup_style_character/{run_id}/{case.case_id}",
        "images": ["10", 0],
    }
    return prompt


def _write_masks(run_id: str) -> tuple[Path, Path]:
    """Write exact hard left/right masks under unique names."""

    paths = tuple(
        COMFY_ROOT / "input" / f"{run_id.lower()}__style-character-{index:02d}.png"
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
