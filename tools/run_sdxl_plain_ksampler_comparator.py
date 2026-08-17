# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run one ordinary SDXL KSampler pink-haired performance comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_model_links import (
    ManagedComfyModelLinks,
    ManagedModelLink,
)
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_couple_parity.cases import load_parity_case
from tools.sdxl_attention_coupling_integration.comfy_model_root import (
    resolve_active_comfy_model_root,
)
from tools.sdxl_attention_coupling_integration.visual_adapter_selections import (
    CHECKPOINT_SELECTION,
)
from tools.sdxl_attention_coupling_integration.visual_inventory import (
    SdxlVisualInventory,
)
from tools.sdxl_full_strength_lora_fidelity.plain_workflow import (
    build_plain_ksampler_workflow,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\universal-regional-adapter\plain-ksampler"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute and preserve one ordinary no-LoRA comparator."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--prompt-case", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = _execute(artifacts, args.inventory, args.prompt_case, args.comfy_root)
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Plain KSampler comparator failed at %s", artifacts.root)
        return 1
    LOGGER.info("Plain KSampler comparator completed: %s", result)
    return 0


def _execute(
    artifacts: IntegrationArtifacts,
    inventory_path: Path,
    prompt_path: Path,
    comfy_root: Path,
) -> Path:
    """Run one measured graph with exact managed lifecycle."""

    inventory = SdxlVisualInventory.load(inventory_path)
    prompts = load_parity_case(prompt_path)
    negative_g = _without_solo_rejections(prompts.base_negative_g)
    negative_l = _without_solo_rejections(prompts.base_negative_l)
    workflow = build_plain_ksampler_workflow(
        run_id=artifacts.run_id,
        checkpoint_name=CHECKPOINT_SELECTION,
        positive_g=prompts.left_positive_g,
        positive_l=prompts.left_positive_l,
        negative_g=negative_g,
        negative_l=negative_l,
    )
    links = ManagedComfyModelLinks(
        model_root=resolve_active_comfy_model_root(comfy_root),
        links=(
            ManagedModelLink(
                inventory.checkpoint_source,
                "checkpoints",
                CHECKPOINT_SELECTION,
            ),
        ),
    )
    with links:
        with ManagedComfyServer(
            comfy_root=comfy_root,
            artifacts=artifacts,
            required_node_ids=workflow.required_node_ids,
            readiness_timeout=240.0,
            launch_arguments=(
                "--disable-all-custom-nodes",
                "--whitelist-custom-nodes",
                "SimpleSyrup",
                "SimpleSyrupBenchmarkProbe",
                "substitute-backend",
            ),
        ) as running:
            started_at = time.perf_counter()
            prompt_id = running.client.submit(workflow.prompt)
            history = running.client.wait_for_history(prompt_id, timeout=1200.0)
            wall_runtime_ms = (time.perf_counter() - started_at) * 1000.0
            reference = extract_saved_image(history, workflow.save_node_id)
            image_bytes = running.client.download_image(reference)
            metrics = _metrics(history, workflow.metrics_node_id)
            image_path = artifacts.root / "plain-pink-control.png"
            image_path.write_bytes(image_bytes)
            result = artifacts.root / "plain-ksampler-result.json"
            result.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "label": "Plain KSampler — pink-haired control — no LoRA",
                        "prompt_id": prompt_id,
                        "wall_runtime_ms": wall_runtime_ms,
                        "metrics": metrics,
                        "image_file": image_path.name,
                        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            port = running.port
            process = running.process
        artifacts.record_cleanup(
            process_running=process.is_running,
            port_available=is_loopback_port_available(port),
        )
    if not links.cleaned:
        raise RuntimeError("Plain KSampler checkpoint link did not clean up.")
    return result


def _metrics(history: JsonObject, node_id: str) -> JsonObject:
    """Decode one exact benchmark metrics object."""

    outputs = history.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(node_id), dict):
        raise ValueError("Plain KSampler history is missing metrics.")
    values = outputs[node_id].get("benchmark_metrics")
    if (
        not isinstance(values, list)
        or len(values) != 1
        or not isinstance(values[0], dict)
    ):
        raise ValueError("Plain KSampler metrics are invalid.")
    return cast(JsonObject, values[0])


def _without_solo_rejections(prompt: str) -> str:
    """Remove exact two-person-only negative tokens."""

    return ", ".join(
        segment.strip()
        for segment in prompt.split(",")
        if segment.strip().casefold() not in {"1girl", "solo"}
    )


if __name__ == "__main__":
    raise SystemExit(main())
