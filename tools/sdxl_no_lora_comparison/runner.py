# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Coordinate one warmed same-process SDXL no-LoRA comparison."""

from __future__ import annotations

from pathlib import Path

from tools.comfy_api import JsonObject, LoopbackComfyClient
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_coupling_integration.comfy_model_root import (
    resolve_active_comfy_model_root,
)
from tools.sdxl_attention_coupling_integration.managed_model_links import (
    ManagedModelLink,
    ManagedSdxlVisualModelLinks,
)
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    CHECKPOINT_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    VisualMaskProfile,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_launch import (
    sdxl_visual_sampling_launch_arguments,
)
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_attention_coupling_integration.visual_masks import (
    ManagedSdxlVisualMasks,
)

from .measurement import SdxlNoLoraMeasurementRecorder
from .results import SdxlNoLoraResultRecorder
from .workflow import build_no_lora_comparison_workflows, without_image_outputs


def run_no_lora_comparison(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
    comfy_root: Path,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
) -> Path:
    """Warm both graph families and record one measured image from each."""

    links = ManagedSdxlVisualModelLinks(
        model_root=resolve_active_comfy_model_root(comfy_root),
        links=(
            ManagedModelLink(
                inventory.checkpoint_source,
                "checkpoints",
                CHECKPOINT_SELECTION,
            ),
        ),
    )
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    system_stats: JsonObject = {}
    with links:
        with masks:
            warmup = build_no_lora_comparison_workflows(
                run_id=f"{artifacts.run_id}-warmup",
                checkpoint_name=CHECKPOINT_SELECTION,
                mask_names=masks.names(VisualMaskProfile.HARD),
                prompts=prompts,
            )
            measured = build_no_lora_comparison_workflows(
                run_id=f"{artifacts.run_id}-measured",
                checkpoint_name=CHECKPOINT_SELECTION,
                mask_names=masks.names(VisualMaskProfile.HARD),
                prompts=prompts,
            )
            required = warmup.required_node_ids | measured.required_node_ids
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=required,
                readiness_timeout=readiness_timeout,
                launch_arguments=sdxl_visual_sampling_launch_arguments(),
            ) as running:
                system_stats = running.system_stats
                _warm(
                    running.client,
                    without_image_outputs(warmup.plain.prompt),
                    prompt_timeout,
                )
                _warm(
                    running.client,
                    without_image_outputs(warmup.regional.prompt),
                    prompt_timeout,
                )
                measurement = SdxlNoLoraMeasurementRecorder(artifacts)
                plain = measurement.measure_plain(
                    running.client,
                    workflow=measured.plain,
                    prompt_timeout=prompt_timeout,
                )
                regional = measurement.measure_regional(
                    running.client,
                    workflow=measured.regional,
                    prompt_timeout=prompt_timeout,
                )
                port = running.port
                process = running.process
            port_available = is_loopback_port_available(port)
            server_cleanup = not process.is_running and port_available
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=port_available,
            )
    if not all((server_cleanup, links.cleaned, masks.cleaned)):
        raise RuntimeError("No-LoRA comparison did not clean every managed owner.")
    return SdxlNoLoraResultRecorder(artifacts.root).record(
        workflows=measured,
        prompts=prompts,
        plain=plain,
        regional=regional,
        system_stats=system_stats,
        port=port,
    )


def _warm(
    client: LoopbackComfyClient,
    prompt: dict[str, JsonObject],
    timeout: float,
) -> None:
    """Execute one image-free warmup graph to stabilize the selected path."""

    prompt_id = client.submit(prompt)
    client.wait_for_history(prompt_id, timeout=timeout)
