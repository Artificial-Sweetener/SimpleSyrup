# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the bounded standard-UNet Attention Couple parity comparison."""

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
from tools.comfy_integration.managed_model_links import (
    ManagedComfyModelLinks,
    ManagedModelLink,
)
from tools.comfy_integration.managed_server import (
    ManagedComfyServer,
    RunningManagedComfy,
)
from tools.sdxl_attention_couple_parity.cases import (
    SdxlAttentionCoupleParityCase,
    load_parity_case,
    parity_case,
)
from tools.sdxl_attention_couple_parity.global_style import ParityGlobalStyle
from tools.sdxl_attention_couple_parity.results import ParityResultRecorder
from tools.sdxl_attention_couple_parity.workflow import (
    BuiltParityWorkflow,
    ParityBackend,
    build_parity_workflow,
)
from tools.sdxl_attention_coupling_integration.comfy_model_root import (
    resolve_active_comfy_model_root,
)
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    CHECKPOINT_SELECTION,
    STYLE_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_case_model import (
    VisualMaskProfile,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_attention_coupling_integration.visual_masks import (
    ManagedSdxlVisualMasks,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\ra01-parity"
)
_CUSTOM_NODE_WHITELIST = ("SimpleSyrup", "substitute-backend", "comfyui-ppm")


def execute_parity_comparison(
    artifacts: IntegrationArtifacts,
    *,
    inventory: SdxlVisualInventory,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
    case: SdxlAttentionCoupleParityCase | None = None,
    global_style: bool = False,
    backends: tuple[ParityBackend, ...] = tuple(ParityBackend),
) -> Path:
    """Run exactly reference then candidate and persist every control artifact."""

    style = (
        ParityGlobalStyle(
            STYLE_SELECTION,
            0.65,
            inventory.style.prompt_g,
            inventory.style.prompt_l,
        )
        if global_style
        else None
    )
    recorder = ParityResultRecorder(
        artifacts.root,
        backends=backends,
        global_style=style,
    )
    links = [
        ManagedModelLink(
            inventory.checkpoint_source,
            "checkpoints",
            CHECKPOINT_SELECTION,
        )
    ]
    if style is not None:
        links.append(ManagedModelLink(inventory.style.source, "loras", STYLE_SELECTION))
    model_links = ManagedComfyModelLinks(
        model_root=resolve_active_comfy_model_root(comfy_root),
        links=tuple(links),
    )
    masks = ManagedSdxlVisualMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    server_cleanup = False
    system_stats: JsonObject = {}
    mask_evidence: tuple[dict[str, object], ...] = ()
    with model_links:
        with masks:
            mask_names = masks.names(VisualMaskProfile.HARD)
            selected_case = case or parity_case()
            workflows = tuple(
                build_parity_workflow(
                    backend=backend,
                    run_id=artifacts.run_id,
                    checkpoint_name=CHECKPOINT_SELECTION,
                    mask_names=mask_names,
                    case=selected_case,
                    global_style=style,
                )
                for backend in backends
            )
            required = frozenset(
                node_id
                for workflow in workflows
                for node_id in workflow.required_node_ids
            )
            mask_evidence = tuple(
                item
                for item in masks.evidence()
                if item["profile"] == VisualMaskProfile.HARD.value
            )
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=required,
                readiness_timeout=readiness_timeout,
                launch_arguments=_launch_arguments(),
            ) as running:
                system_stats = running.system_stats
                for workflow in workflows:
                    image_path = _execute_workflow(
                        workflow,
                        running=running,
                        recorder=recorder,
                        prompt_timeout=prompt_timeout,
                    )
                    LOGGER.info(
                        "Preserved %s parity artifact: %s",
                        workflow.backend.value,
                        image_path,
                    )
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
        mask_cleanup=masks.cleaned,
        mask_evidence=mask_evidence,
    )


def _execute_workflow(
    workflow: BuiltParityWorkflow,
    *,
    running: RunningManagedComfy,
    recorder: ParityResultRecorder,
    prompt_timeout: float,
) -> Path:
    """Submit one backend graph and immediately preserve its labeled image."""

    started_at = time.perf_counter()
    prompt_id = running.client.submit(workflow.prompt)
    history = running.client.wait_for_history(prompt_id, timeout=prompt_timeout)
    wall_runtime_ms = (time.perf_counter() - started_at) * 1000.0
    reference = extract_saved_image(history, workflow.save_node_id)
    image_bytes = running.client.download_image(reference)
    return recorder.record(
        workflow,
        history=history,
        prompt_id=prompt_id,
        image_bytes=image_bytes,
        wall_runtime_ms=wall_runtime_ms,
    )


def _launch_arguments() -> tuple[str, ...]:
    """Load only the owners needed by the isolated parity process."""

    return (
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        *_CUSTOM_NODE_WHITELIST,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command arguments and return one process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=240.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    parser.add_argument("--global-style", action="store_true")
    parser.add_argument(
        "--backend",
        action="append",
        type=ParityBackend,
        choices=tuple(ParityBackend),
        default=[],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = execute_parity_comparison(
            artifacts,
            inventory=SdxlVisualInventory.load(args.inventory),
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
            case=load_parity_case(args.case) if args.case is not None else None,
            global_style=args.global_style,
            backends=tuple(args.backend) if args.backend else tuple(ParityBackend),
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception(
            "Managed SDXL Attention Couple parity failed at %s", artifacts.root
        )
        return 1
    LOGGER.info("Managed SDXL Attention Couple parity completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
