"""Run one fixed-seed CHARACTER_A regional-composition proof in managed ComfyUI."""

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
    / "user-prompt-character_a-ownership-proof"
)
SOURCE_ROOT = (
    COMFY_ROOT
    / "benchmark_artifacts"
    / "anima-regional-prompting-v1"
    / "user-prompt-character_a-character-lora-pair"
    / "20260812T163713Z-219c5384"
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
            image_bytes = running.client.download_image(reference)
            artifacts.record_success(
                system_stats=running.system_stats,
                workflow=prompt,
                history=history,
                prompt_id=prompt_id,
                image_reference=reference,
                image_bytes=image_bytes,
            )
        artifacts.record_cleanup(
            process_running=running.process.is_running,
            port_available=is_loopback_port_available(running.port),
        )
        _label(artifacts.root)
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("CHARACTER_A ownership proof failed at %s", artifacts.root)
        return 1
    finally:
        for path in mask_paths:
            path.unlink(missing_ok=True)
    LOGGER.info("CHARACTER_A ownership proof completed at %s", artifacts.root)
    return 0


def _source_prompt() -> dict[str, JsonObject]:
    """Load the exact prior failed CHARACTER_A workflow as the fixed-seed source."""

    decoded = json.loads(
        (SOURCE_ROOT / "character_a-lora-right-only__workflow.json").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(decoded, dict):
        raise TypeError("CHARACTER_A source workflow must be a JSON object.")
    return cast(dict[str, JsonObject], decoded)


def _write_masks(run_id: str) -> tuple[Path, Path]:
    """Write exact hard left/right 1024-square masks under unique names."""

    paths = (
        COMFY_ROOT / "input" / f"{run_id.lower()}__character_a-region-00.png",
        COMFY_ROOT / "input" / f"{run_id.lower()}__character_a-region-01.png",
    )
    for index, path in enumerate(paths):
        image = Image.new("RGB", (1024, 1024), color="black")
        draw = ImageDraw.Draw(image)
        draw.rectangle(
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
    prompt["5"]["inputs"] = {
        "model": ["2", 0],
        "run_id": f"{run_id}:character_a-ownership:metrics",
    }
    prompt["6"]["inputs"] = {
        "model": ["5", 0],
        "run_id": f"{run_id}:character_a-ownership:regional-diagnostics",
    }
    prompt["8"]["inputs"] = {
        "latent": ["7", 0],
        "run_id": f"{run_id}:character_a-ownership:metrics",
    }
    prompt["9"]["inputs"] = {
        "latent": ["8", 0],
        "run_id": f"{run_id}:character_a-ownership:regional-diagnostics",
    }
    prompt["11"]["inputs"] = {
        "filename_prefix": f"simple_syrup_character_a_ownership/{run_id}/character_a-right-only",
        "images": ["10", 0],
    }


def _label(root: Path) -> None:
    """Place failed baseline and corrected output side by side at full resolution."""

    before = Image.open(
        SOURCE_ROOT / "character_a-lora-right-only__seed-3141592653.png"
    ).convert("RGB")
    after = Image.open(root / "baseline.png").convert("RGB")
    if before.size != (1024, 1024) or after.size != (1024, 1024):
        raise ValueError(
            "CHARACTER_A proof images must retain original 1024-square resolution."
        )
    header = 96
    canvas = Image.new("RGB", (2048, 1024 + header), color=(20, 20, 20))
    canvas.paste(before, (0, header))
    canvas.paste(after, (1024, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(r"<SYSTEM_FONT>", 28)
    draw.text(
        (24, 30), "BEFORE — global self-attention takeover", fill="white", font=font
    )
    draw.text(
        (1048, 30), "AFTER — phased regional composition", fill="white", font=font
    )
    canvas.save(root / "character_a-ownership__labeled-comparison.png", format="PNG")


if __name__ == "__main__":
    raise SystemExit(main())
