# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run one real incoming Anima baseline in an isolated managed Comfy server."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.baseline_workflow import build_baseline_workflow
from tools.comfy_integration.default_paths import (
    default_benchmark_artifact_root,
    default_comfy_root,
)
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = default_comfy_root()
DEFAULT_OUTPUT_ROOT = default_benchmark_artifact_root(
    "anima-regional-prompting-v1/p0.9"
)


def execute_baseline(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> None:
    """Execute, capture, and clean one managed baseline run."""

    workflow = build_baseline_workflow(artifacts.run_id)
    with ManagedComfyServer(
        comfy_root=comfy_root,
        artifacts=artifacts,
        required_node_ids=workflow.required_node_ids,
        readiness_timeout=readiness_timeout,
    ) as running:
        prompt_id = running.client.submit(workflow.prompt)
        history = running.client.wait_for_history(prompt_id, timeout=prompt_timeout)
        image_reference = extract_saved_image(history, workflow.save_node_id)
        image_bytes = running.client.download_image(image_reference)
        artifacts.record_success(
            system_stats=running.system_stats,
            workflow=workflow.prompt,
            history=history,
            prompt_id=prompt_id,
            image_reference=image_reference,
            image_bytes=image_bytes,
        )
    artifacts.record_cleanup(
        process_running=running.process.is_running,
        port_available=is_loopback_port_available(running.port),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=180.0)
    parser.add_argument("--prompt-timeout", type=float, default=600.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        execute_baseline(
            artifacts,
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception(
            "Managed baseline failed; evidence retained at %s", artifacts.root
        )
        return 1
    LOGGER.info("Managed baseline completed at %s", artifacts.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
