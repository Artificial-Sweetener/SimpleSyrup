# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run one fixed-seed PRIMARY_ADAPTER regional-composition proof in managed ComfyUI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

from PIL import Image, ImageDraw, ImageFont

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer

LOGGER = logging.getLogger(__name__)
COMFY_ROOT = Path(r"<COMFY_ROOT>")
OUTPUT_ROOT = (
    COMFY_ROOT
    / "benchmark_artifacts"
    / "anima-regional-prompting-v1"
    / "user-prompt-primary_adapter-ownership-proof"
)
SOURCE_ROOT = (
    COMFY_ROOT
    / "benchmark_artifacts"
    / "anima-regional-prompting-v1"
    / "user-prompt-reference_checkpoint-regional-lora-pair"
    / "20260812T160702Z-ad444583"
)


def main() -> int:
    """Execute, persist, label, and clean one exact visual proof."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(OUTPUT_ROOT)
    mask_paths: tuple[Path, ...] = ()
    try:
        prompt = _source_prompt()
        mask_paths = _write_masks(artifacts.run_id)
        _specialize(prompt, artifacts.run_id, mask_paths)
        required = frozenset(cast(str, node["class_type"]) for node in prompt.values())
        with ManagedComfyServer(
            comfy_root=COMFY_ROOT,
            artifacts=artifacts,
            required_node_ids=required,
            readiness_timeout=240.0,
        ) as running:
            prompt_id = running.client.submit(prompt)
            history = running.client.wait_for_history(prompt_id, timeout=1200.0)
            reference = extract_saved_image(history, "11")
            artifacts.record_success(
                system_stats=running.system_stats,
                workflow=prompt,
                history=history,
                prompt_id=prompt_id,
                image_reference=reference,
                image_bytes=running.client.download_image(reference),
            )
        artifacts.record_cleanup(
            process_running=running.process.is_running,
            port_available=is_loopback_port_available(running.port),
        )
        _label(artifacts.root)
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception(
            "PRIMARY_ADAPTER composition proof failed at %s", artifacts.root
        )
        return 1
    finally:
        for path in mask_paths:
            path.unlink(missing_ok=True)
    LOGGER.info("PRIMARY_ADAPTER composition proof completed at %s", artifacts.root)
    return 0


def _source_prompt() -> dict[str, JsonObject]:
    """Load the prior selected workflow as the fixed-seed source."""

    decoded = json.loads(
        (
            SOURCE_ROOT
            / "reference_checkpoint-primary_adapter-pink-only__workflow.json"
        ).read_text(encoding="utf-8")
    )
    if not isinstance(decoded, dict):
        raise TypeError("PRIMARY_ADAPTER source workflow must be a JSON object.")
    return cast(dict[str, JsonObject], decoded)


def _write_masks(run_id: str) -> tuple[Path, Path]:
    """Write exact hard left/right 1024-square masks under unique names."""

    paths = (
        COMFY_ROOT / "input" / f"{run_id.lower()}__primary_adapter-region-00.png",
        COMFY_ROOT / "input" / f"{run_id.lower()}__primary_adapter-region-01.png",
    )
    for index, path in enumerate(paths):
        image = Image.new("RGB", (1024, 1024), color="black")
        ImageDraw.Draw(image).rectangle(
            (0 if index == 0 else 512, 0, 511 if index == 0 else 1023, 1023),
            fill="white",
        )
        image.save(path, format="PNG", compress_level=9)
    return paths


def _specialize(
    prompt: dict[str, JsonObject],
    run_id: str,
    masks: tuple[Path, ...],
) -> None:
    """Replace only run-local mask, diagnostic, and output identities."""

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
            "run_id": f"{run_id}:primary_adapter-ownership:{suffix}",
        }
    prompt["8"]["inputs"] = {
        "latent": ["7", 0],
        "run_id": f"{run_id}:primary_adapter-ownership:metrics",
    }
    prompt["9"]["inputs"] = {
        "latent": ["8", 0],
        "run_id": f"{run_id}:primary_adapter-ownership:regional-diagnostics",
    }
    prompt["11"]["inputs"] = {
        "filename_prefix": (
            f"simple_syrup_adapter_ownership/{run_id}/adapter-left-only"
        ),
        "images": ["10", 0],
    }


def _label(root: Path) -> None:
    """Show the baseline, takeover, and phased result at full resolution."""

    inputs = (
        SOURCE_ROOT / "reference_checkpoint-no-lora__seed-3141592653.png",
        SOURCE_ROOT
        / "reference_checkpoint-primary_adapter-pink-only__seed-3141592653.png",
        root / "baseline.png",
    )
    labels = (
        "NO LORA — reference checkpoint baseline",
        "BEFORE — PRIMARY_ADAPTER left, global image attention",
        "AFTER — phased regional composition",
    )
    header = 96
    canvas = Image.new("RGB", (3072, 1024 + header), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(r"<SYSTEM_FONT>", 26)
    for index, (path, label) in enumerate(zip(inputs, labels, strict=True)):
        image = Image.open(path).convert("RGB")
        if image.size != (1024, 1024):
            raise ValueError("PRIMARY_ADAPTER proof images must remain 1024 square.")
        canvas.paste(image, (index * 1024, header))
        draw.text((index * 1024 + 20, 32), label, fill="white", font=font)
    canvas.save(
        root / "primary_adapter-ownership__labeled-comparison.png", format="PNG"
    )


if __name__ == "__main__":
    raise SystemExit(main())
