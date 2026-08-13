# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run the complete P9.4 text-encoder LoRA matrix in managed ComfyUI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from tools.anima_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as FULL_NODE_ID,
)
from tools.anima_attention_coupling_integration.schema import (
    validate_public_node_metadata as validate_full_metadata,
)
from tools.anima_attention_coupling_workflow import HEIGHT, WIDTH
from tools.anima_contextual_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as CONTEXTUAL_NODE_ID,
)
from tools.anima_contextual_attention_coupling_integration.matrix import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from tools.anima_contextual_attention_coupling_integration.schema import (
    validate_public_node_metadata as validate_contextual_metadata,
)
from tools.anima_tiled_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID as TILED_NODE_ID,
)
from tools.anima_tiled_attention_coupling_integration.schema import (
    validate_public_node_metadata as validate_tiled_metadata,
)
from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.text_encoder_lora_integration.fixture import (
    PINNED_TEXT_ENCODER_LORA_FIXTURE,
    validate_text_encoder_lora_fixture,
)
from tools.text_encoder_lora_integration.matrix import (
    MASK_CASE_ID,
    TextEncoderLoraSpatialMode,
    cases,
)
from tools.text_encoder_lora_integration.results import (
    TextEncoderLoraResultRecorder,
)
from tools.text_encoder_lora_integration.workflow import (
    TextEncoderLoraWorkflowBuilder,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p9.4"
)


def execute_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Execute every full, tiled, and Contextual P9.4 workflow."""

    fixture_identity = validate_text_encoder_lora_fixture(
        PINNED_TEXT_ENCODER_LORA_FIXTURE
    )
    definitions = cases()
    manifest_case = next(
        case for case in load_manifest().cases if case.case_id == MASK_CASE_ID
    )
    input_root = comfy_root / "input"
    full_writer = MaskArtifactWriter(input_root, f"p94-full-{artifacts.run_id.lower()}")
    upscale_writer = MaskArtifactWriter(
        input_root,
        f"p94-upscale-{artifacts.run_id.lower()}",
    )
    full_masks = full_writer.write_case(manifest_case, width=WIDTH, height=HEIGHT)
    upscale_masks = upscale_writer.write_case(
        manifest_case,
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
    )
    mask_paths = tuple(
        (input_root / name).resolve() for name in (*full_masks, *upscale_masks)
    )
    builder = TextEncoderLoraWorkflowBuilder()
    workflows = tuple(
        builder.build(
            case,
            run_id=f"{artifacts.run_id}-{case.case_id}",
            mask_names=(
                full_masks
                if case.spatial_mode is TextEncoderLoraSpatialMode.FULL
                else upscale_masks
            ),
        )
        for case in definitions
    )
    required = frozenset().union(
        *(workflow.required_node_ids for workflow in workflows)
    )
    recorder = TextEncoderLoraResultRecorder(artifacts.root)
    recorder.record_fixture(fixture_identity)
    try:
        with ManagedComfyServer(
            comfy_root=comfy_root,
            artifacts=artifacts,
            required_node_ids=required,
            readiness_timeout=readiness_timeout,
        ) as running:
            metadata_validators = (
                (FULL_NODE_ID, validate_full_metadata),
                (TILED_NODE_ID, validate_tiled_metadata),
                (CONTEXTUAL_NODE_ID, validate_contextual_metadata),
            )
            for node_id, validator in metadata_validators:
                metadata = running.client.node_metadata(node_id)
                validator(metadata)
                recorder.record_metadata(node_id, metadata)
            for case, workflow in zip(definitions, workflows, strict=True):
                LOGGER.info("P9.4 starting %s — %s", case.case_id, case.label)
                started = time.perf_counter()
                prompt_id = running.client.submit(workflow.prompt)
                history = running.client.wait_for_history(
                    prompt_id,
                    timeout=prompt_timeout,
                )
                wall_runtime_ms = (time.perf_counter() - started) * 1000.0
                reference = extract_saved_image(history, workflow.save_node_id)
                image = running.client.download_image(reference)
                path = recorder.record_case(
                    case,
                    workflow,
                    prompt_id=prompt_id,
                    history=history,
                    reference=reference,
                    image_bytes=image,
                    wall_runtime_ms=wall_runtime_ms,
                )
                LOGGER.info("P9.4 accepted %s", path)
            system_stats = running.system_stats
            port = running.port
            process = running.process
    finally:
        for mask_path in mask_paths:
            mask_path.unlink(missing_ok=True)
    masks_removed = not any(path.exists() for path in mask_paths)
    port_available = is_loopback_port_available(port)
    cleanup_verified = not process.is_running and port_available
    artifacts.record_cleanup(
        process_running=process.is_running,
        port_available=port_available,
    )
    return recorder.finalize(
        definitions,
        system_stats=system_stats,
        cleanup_verified=cleanup_verified,
        masks_removed=masks_removed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return one process status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--prompt-timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = execute_matrix(
            artifacts,
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("P9.4 integration failed at %s", artifacts.root)
        return 1
    LOGGER.info("P9.4 integration completed at %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
