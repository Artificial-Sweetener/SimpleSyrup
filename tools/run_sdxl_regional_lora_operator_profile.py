# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Profile one warmed SDXL regional-LoRA denoiser call without image output."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_couple_parity.cases import load_parity_case
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
from tools.sdxl_full_strength_lora_fidelity.composition_cases import (
    full_strength_composition_cases,
)
from tools.sdxl_regional_lora_performance.indexed_profile_workflow import (
    decode_indexed_operator_profile,
)
from tools.sdxl_regional_lora_performance.operator_profile_workflow import (
    build_sdxl_operator_profile_workflow,
)
from tools.sdxl_regional_lora_performance.two_adapter_workflow import (
    TwoAdapterPerformanceMode,
    build_two_adapter_performance_workflow,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\sdxl-operator-profile"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Warm the locked regional graph and capture its final model call."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prompt-case", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=Path(r"<COMFY_ROOT>"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in TwoAdapterPerformanceMode),
        default=TwoAdapterPerformanceMode.REGIONAL.value,
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = _run(
            artifacts,
            inventory=SdxlVisualInventory.load(args.inventory),
            prompt_case=args.prompt_case,
            comfy_root=args.comfy_root,
            mode=TwoAdapterPerformanceMode(args.mode),
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("SDXL operator profile failed at %s", artifacts.root)
        return 1
    LOGGER.info("SDXL operator profile completed: %s", result)
    return 0


def _run(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    prompt_case: Path,
    comfy_root: Path,
    mode: TwoAdapterPerformanceMode,
) -> Path:
    """Execute one managed warmup and one final-call profile trajectory."""

    prompts = load_parity_case(prompt_case)
    prompt_set = SdxlVisualPromptSet(
        base_positive_g=prompts.base_positive_g,
        base_positive_l=prompts.base_positive_l,
        base_negative_g=prompts.base_negative_g,
        base_negative_l=prompts.base_negative_l,
        left_positive_g=prompts.left_positive_g,
        left_positive_l=prompts.left_positive_l,
        right_positive_g=prompts.right_positive_g,
        right_positive_l=prompts.right_positive_l,
        left_negative_g=prompts.left_negative_g,
        left_negative_l=prompts.left_negative_l,
        right_negative_g=prompts.right_negative_g,
        right_negative_l=prompts.right_negative_l,
    )
    case = full_strength_composition_cases(inventory, prompt_set)[2]
    trace_path = artifacts.root / "steady-denoiser-trace.json"
    links = build_sdxl_visual_model_links(comfy_root, inventory)
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    with links:
        with masks:
            mask_names = masks.names(case.mask_profile)
            warmup = build_two_adapter_performance_workflow(
                run_id=f"{artifacts.run_id}:warmup",
                checkpoint_name=CHECKPOINT_SELECTION,
                mask_names=mask_names,
                case=case,
                mode=mode,
            )
            profiled = build_sdxl_operator_profile_workflow(
                run_id=f"{artifacts.run_id}:profiled",
                checkpoint_name=CHECKPOINT_SELECTION,
                mask_names=mask_names,
                case=case,
                trace_path=trace_path,
                mode=mode,
                call_index=30,
            )
            required = warmup.required_node_ids | profiled.required_node_ids
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=required,
                readiness_timeout=240.0,
                launch_arguments=sdxl_visual_sampling_launch_arguments(),
            ) as running:
                warmup_id = running.client.submit(warmup.prompt)
                running.client.wait_for_history(warmup_id, timeout=1200.0)
                profile_id = running.client.submit(profiled.prompt)
                history = running.client.wait_for_history(profile_id, timeout=1200.0)
                capture = decode_indexed_operator_profile(
                    history,
                    profiled.profile_node_id,
                )
                (artifacts.root / "profile-history.json").write_text(
                    json.dumps(history, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                (artifacts.root / "profile-workflow.json").write_text(
                    json.dumps(profiled.prompt, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                result = artifacts.root / "operator-profile.json"
                result.write_text(
                    json.dumps(capture, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                port = running.port
                process = running.process
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
    if not links.cleaned or not masks.cleaned:
        raise RuntimeError("SDXL operator profile external cleanup failed.")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
