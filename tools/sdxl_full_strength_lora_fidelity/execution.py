# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute the bounded full-strength LoRA reference-parity matrix."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    CHECKPOINT_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_launch import (
    sdxl_visual_sampling_launch_arguments,
)
from tools.sdxl_attention_coupling_integration.visual_matrix_execution import (
    build_sdxl_visual_model_links,
)

from .cases import FullStrengthFidelityCase
from .history import decode_fidelity_history
from .mask import ManagedAllOneMask
from .results import FullStrengthFidelityResultRecorder
from .workflow import build_fidelity_workflow

LOGGER = logging.getLogger(__name__)


def execute_fidelity_cases(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    cases: tuple[FullStrengthFidelityCase, ...],
    comfy_root: Path,
    base_positive_g: str,
    base_positive_l: str,
    negative_g: str,
    negative_l: str,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Run each paired case once in one managed nonstandard-port server."""

    recorder = FullStrengthFidelityResultRecorder(artifacts.root, cases=cases)
    model_links = build_sdxl_visual_model_links(comfy_root, inventory)
    mask = ManagedAllOneMask(input_root=comfy_root / "input", run_id=artifacts.run_id)
    system_stats: JsonObject = {}
    server_cleanup = False
    with model_links:
        with mask:
            workflows = tuple(
                (
                    case,
                    build_fidelity_workflow(
                        run_id=artifacts.run_id,
                        checkpoint_name=CHECKPOINT_SELECTION,
                        mask_name=mask.name,
                        base_positive_g=base_positive_g,
                        base_positive_l=base_positive_l,
                        negative_g=negative_g,
                        negative_l=negative_l,
                        case=case,
                    ),
                )
                for case in cases
            )
            required = frozenset(
                node_id
                for _case, workflow in workflows
                for node_id in workflow.required_node_ids
            )
            mask_evidence = mask.evidence()
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=required,
                readiness_timeout=readiness_timeout,
                launch_arguments=sdxl_visual_sampling_launch_arguments(),
            ) as running:
                system_stats = running.system_stats
                for case, workflow in workflows:
                    LOGGER.info("Generating labeled artifact: %s", case.label)
                    started_at = time.perf_counter()
                    try:
                        prompt_id = running.client.submit(workflow.prompt)
                        history = running.client.wait_for_history(
                            prompt_id, timeout=prompt_timeout
                        )
                        evidence = decode_fidelity_history(history, workflow)
                        image_bytes = running.client.download_image(
                            evidence.image_reference
                        )
                        image_path = recorder.record_case(
                            workflow,
                            case=case,
                            history=history,
                            prompt_id=prompt_id,
                            evidence=evidence,
                            image_bytes=image_bytes,
                            wall_runtime_ms=(time.perf_counter() - started_at) * 1000.0,
                        )
                        LOGGER.info("Labeled artifact ready: %s", image_path)
                    except BaseException as error:
                        recorder.record_failure(case, error)
                        raise
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
        mask_cleanup=mask.cleaned,
        mask_evidence=mask_evidence,
    )
