# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Capture zero-use and one-use indexed SDXL operator profiles."""

from __future__ import annotations

import json
from pathlib import Path

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
from tools.sdxl_attention_coupling_integration.visual_lora_baseline_cases import (
    SdxlVisualPromptSet,
)
from tools.sdxl_attention_coupling_integration.visual_masks import (
    ManagedSdxlVisualMasks,
)
from tools.sdxl_attention_coupling_integration.visual_matrix_execution import (
    build_sdxl_visual_model_links,
)

from .indexed_profile_workflow import (
    BuiltIndexedOperatorProfileWorkflow,
    build_indexed_operator_profile_workflow,
    decode_indexed_operator_profile,
)
from .scaling_cases import (
    DeclaredSdxlRegionalScalingCase,
    SdxlRegionalScalingProfile,
    regional_scaling_cases,
)
from .scaling_workflow import build_regional_scaling_steady_state_workflow
from .steady_state_workflow import BuiltSdxlSteadyStateWorkflow


def run_scaling_profiles(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    prompts: SdxlVisualPromptSet,
    comfy_root: Path,
    readiness_timeout: float = 240.0,
    prompt_timeout: float = 1200.0,
    call_index: int = 30,
) -> Path:
    """Warm and profile one indexed zero-use and one-use denoiser call."""

    if (
        isinstance(call_index, bool)
        or not isinstance(call_index, int)
        or call_index < 1
    ):
        raise ValueError("SDXL scaling profile call index must be positive.")

    declared = regional_scaling_cases(
        prompts,
        left_trigger_g=inventory.left_character.prompt_g,
        left_trigger_l=inventory.left_character.prompt_l,
        right_trigger_g=inventory.right_character.prompt_g,
        right_trigger_l=inventory.right_character.prompt_l,
        style_trigger_g=inventory.style.prompt_g,
        style_trigger_l=inventory.style.prompt_l,
    )[:2]
    links = build_sdxl_visual_model_links(comfy_root, inventory)
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    index: dict[str, object] = {}
    with links:
        with masks:
            mask_names = masks.names(declared[0].case.mask_profile)
            workflows = tuple(
                _profile_workflows(
                    artifacts=artifacts,
                    declared=item,
                    mask_names=mask_names,
                    call_index=call_index,
                )
                for item in declared
            )
            required = frozenset(
                node_id
                for _, warmup, profiled, _ in workflows
                for node_id in warmup.required_node_ids | profiled.required_node_ids
            )
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=required,
                readiness_timeout=readiness_timeout,
                launch_arguments=sdxl_visual_sampling_launch_arguments(),
            ) as running:
                for item, warmup, profiled, trace_path in workflows:
                    warmup_id = running.client.submit(warmup.prompt)
                    running.client.wait_for_history(warmup_id, timeout=prompt_timeout)
                    profile_id = running.client.submit(profiled.prompt)
                    history = running.client.wait_for_history(
                        profile_id,
                        timeout=prompt_timeout,
                    )
                    capture = decode_indexed_operator_profile(
                        history,
                        profiled.profile_node_id,
                    )
                    index[item.profile.value] = _write_profile_artifacts(
                        artifacts=artifacts,
                        profile=item.profile,
                        profiled=profiled,
                        history=history,
                        capture=capture,
                        trace_path=trace_path,
                        call_index=call_index,
                    )
                port = running.port
                process = running.process
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    if not links.cleaned or not masks.cleaned:
        raise RuntimeError("SDXL scaling profile external cleanup failed.")
    result = artifacts.root / "scaling-profile-index.json"
    result.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _profile_workflows(
    *,
    artifacts: IntegrationArtifacts,
    declared: DeclaredSdxlRegionalScalingCase,
    mask_names: tuple[str, str],
    call_index: int = 30,
) -> tuple[
    DeclaredSdxlRegionalScalingCase,
    BuiltSdxlSteadyStateWorkflow,
    BuiltIndexedOperatorProfileWorkflow,
    Path,
]:
    """Build one warmup and one indexed-call profile graph."""

    warmup = build_regional_scaling_steady_state_workflow(
        checkpoint_name=CHECKPOINT_SELECTION,
        mask_names=mask_names,
        declared=declared,
    )
    profile_source = build_regional_scaling_steady_state_workflow(
        checkpoint_name=CHECKPOINT_SELECTION,
        mask_names=mask_names,
        declared=declared,
    )
    trace_path = artifacts.root / f"{declared.profile.value}-trace.json"
    profiled = build_indexed_operator_profile_workflow(
        prompt=profile_source.prompt,
        sampler_node_id=profile_source.sampler_node_id,
        run_id=f"{artifacts.run_id}:{declared.profile.value}:profile",
        trace_path=trace_path,
        call_index=call_index,
    )
    return declared, warmup, profiled, trace_path


def _write_profile_artifacts(
    *,
    artifacts: IntegrationArtifacts,
    profile: SdxlRegionalScalingProfile,
    profiled: BuiltIndexedOperatorProfileWorkflow,
    history: dict[str, object],
    capture: dict[str, object],
    trace_path: Path,
    call_index: int,
) -> dict[str, object]:
    """Persist one complete profile bundle and return its index entry."""

    stem = profile.value
    workflow_path = artifacts.root / f"{stem}-workflow.json"
    history_path = artifacts.root / f"{stem}-history.json"
    capture_path = artifacts.root / f"{stem}-profile.json"
    workflow_path.write_text(
        json.dumps(profiled.prompt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    capture_path.write_text(
        json.dumps(capture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "call_index": call_index,
        "workflow": str(workflow_path),
        "history": str(history_path),
        "profile": str(capture_path),
        "trace": str(trace_path),
    }
