# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run one native-Comfy SDXL global-LoRA reference image at a time."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_couple_parity.cases import parity_case
from tools.sdxl_attention_coupling_integration.comfy_model_root import (
    resolve_active_comfy_model_root,
)
from tools.sdxl_attention_coupling_integration.managed_model_links import (
    ManagedModelLink,
    ManagedSdxlVisualModelLinks,
)
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    CHECKPOINT_SELECTION,
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_global_lora_reference.cases import NativeGlobalLoraCase
from tools.sdxl_global_lora_reference.results import (
    NativeGlobalLoraResultRecorder,
)
from tools.sdxl_global_lora_reference.workflow import (
    build_native_global_lora_workflow,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\ra05-native-global-lora-reference"
)


def execute_native_global_lora_reference(
    artifacts: IntegrationArtifacts,
    *,
    case: NativeGlobalLoraCase,
    inventory: SdxlVisualInventory,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Run and preserve exactly one ordinary native-Comfy reference case."""

    recorder = NativeGlobalLoraResultRecorder(artifacts.root)
    model_links = ManagedSdxlVisualModelLinks(
        model_root=resolve_active_comfy_model_root(comfy_root),
        links=(
            ManagedModelLink(
                inventory.checkpoint_source,
                "checkpoints",
                CHECKPOINT_SELECTION,
            ),
            ManagedModelLink(inventory.style.source, "loras", STYLE_SELECTION),
        ),
    )
    workflow = build_native_global_lora_workflow(
        case=case,
        run_id=artifacts.run_id,
        checkpoint_name=CHECKPOINT_SELECTION,
        lora_name=STYLE_SELECTION,
        prompt_case=parity_case(),
        style_prompt_g=inventory.style.prompt_g,
        style_prompt_l=inventory.style.prompt_l,
    )
    system_stats: JsonObject = {}
    server_cleanup = False
    with model_links:
        with ManagedComfyServer(
            comfy_root=comfy_root,
            artifacts=artifacts,
            required_node_ids=workflow.required_node_ids,
            readiness_timeout=readiness_timeout,
            launch_arguments=(
                "--disable-all-custom-nodes",
                "--whitelist-custom-nodes",
                "substitute-backend",
            ),
        ) as running:
            system_stats = running.system_stats
            started_at = time.perf_counter()
            prompt_id = running.client.submit(workflow.prompt)
            history = running.client.wait_for_history(
                prompt_id,
                timeout=prompt_timeout,
            )
            wall_runtime_ms = (time.perf_counter() - started_at) * 1000.0
            reference = extract_saved_image(history, workflow.save_node_id)
            image_bytes = running.client.download_image(reference)
            original, labeled = recorder.record(
                workflow,
                history=history,
                prompt_id=prompt_id,
                image_bytes=image_bytes,
                wall_runtime_ms=wall_runtime_ms,
            )
            LOGGER.info("Preserved native original: %s", original)
            LOGGER.info("Preserved labeled native artifact: %s", labeled)
            port = running.port
            process = running.process
        server_cleanup = not process.is_running and is_loopback_port_available(port)
        artifacts.record_cleanup(
            process_running=process.is_running,
            port_available=is_loopback_port_available(port),
        )
    return recorder.finalize(
        system_stats=system_stats,
        server_cleanup=server_cleanup,
        model_cleanup=model_links.cleaned,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one case selection and return its managed execution status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        type=NativeGlobalLoraCase,
        choices=tuple(NativeGlobalLoraCase),
        required=True,
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=240.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = execute_native_global_lora_reference(
            artifacts,
            case=args.case,
            inventory=SdxlVisualInventory.load(args.inventory),
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception(
            "Managed native SDXL global-LoRA reference failed at %s",
            artifacts.root,
        )
        return 1
    LOGGER.info("Managed native SDXL global-LoRA reference completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
