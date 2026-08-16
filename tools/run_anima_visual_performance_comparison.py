# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate and time three visibly labeled Anima comparison images."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tools.anima_visual_performance_comparison.cases import visual_cases
from tools.anima_visual_performance_comparison.inventory import AnimaVisualInventory
from tools.anima_visual_performance_comparison.masks import (
    ManagedAnimaComparisonMasks,
)
from tools.anima_visual_performance_comparison.results import (
    AnimaVisualResultRecorder,
)
from tools.anima_visual_performance_comparison.workflow import (
    build_anima_visual_workflow,
)
from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\visual-performance-fresh"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the exact three-case image and timing matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.mode.value for case in visual_cases()),
        help="Run only the selected case; repeat to select multiple cases.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    selected_cases = frozenset(args.case or ())
    cases = tuple(
        case
        for case in visual_cases()
        if not selected_cases or case.mode.value in selected_cases
    )
    masks = ManagedAnimaComparisonMasks(
        input_root=args.comfy_root / "input",
        run_id=artifacts.run_id,
    )
    recorder = AnimaVisualResultRecorder(artifacts.root)
    try:
        inventory = AnimaVisualInventory.load(args.inventory)
        with masks:
            workflows = tuple(
                build_anima_visual_workflow(
                    run_id=artifacts.run_id,
                    case=case,
                    inventory=inventory,
                    mask_names=masks.names,
                )
                for case in cases
            )
            required = frozenset(
                node for workflow in workflows for node in workflow.required_node_ids
            )
            with ManagedComfyServer(
                comfy_root=args.comfy_root,
                artifacts=artifacts,
                required_node_ids=required,
                readiness_timeout=240.0,
                launch_arguments=(
                    "--disable-all-custom-nodes",
                    "--whitelist-custom-nodes",
                    "SimpleSyrup",
                    "SimpleSyrupBenchmarkProbe",
                    "comfyui-prompt-control",
                    "substitute-backend",
                ),
            ) as running:
                for case, workflow in zip(cases, workflows, strict=True):
                    started = time.perf_counter()
                    prompt_id = running.client.submit(workflow.prompt)
                    history = running.client.wait_for_history(prompt_id, timeout=1800.0)
                    wall_runtime_ms = (time.perf_counter() - started) * 1000.0
                    reference = extract_saved_image(history, workflow.save_node_id)
                    image_bytes = running.client.download_image(reference)
                    metrics = _metrics(history, workflow.metrics_node_id)
                    labeled = recorder.record(
                        case,
                        workflow=workflow.prompt,
                        history=history,
                        image_bytes=image_bytes,
                        metrics=metrics,
                        wall_runtime_ms=wall_runtime_ms,
                    )
                    LOGGER.info("Labeled Anima artifact ready: %s", labeled)
                port = running.port
                process = running.process
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=is_loopback_port_available(port),
            )
        if not masks.cleaned:
            raise RuntimeError("Anima comparison masks did not clean up.")
        result = recorder.finalize(cases)
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Anima visual performance comparison failed")
        return 1
    LOGGER.info("Anima visual performance comparison completed: %s", result)
    return 0


def _metrics(history: JsonObject, node_id: str) -> JsonObject:
    """Decode one exact benchmark metrics object."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(node_id), dict):
        raise ValueError("Anima visual history is missing metrics.")
    values = outputs[node_id].get("benchmark_metrics")
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError("Anima visual metrics are invalid.")
    return cast(JsonObject, values[0])


if __name__ == "__main__":
    raise SystemExit(main())
