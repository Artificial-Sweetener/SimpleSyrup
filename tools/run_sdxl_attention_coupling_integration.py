"""Run the Phase 8 public SDXL Attention Coupling matrix in managed ComfyUI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.sdxl_attention_coupling_integration.checkpoint_link import (
    CheckpointArtifactIdentity,
    ManagedCheckpointLink,
)
from tools.sdxl_attention_coupling_integration.history import decode_sdxl_history
from tools.sdxl_attention_coupling_integration.masks import ManagedSdxlSplitMasks
from tools.sdxl_attention_coupling_integration.matrix import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_SOURCE_NAME,
    CHECKPOINT_STABLE_NAME,
    MODES,
)
from tools.sdxl_attention_coupling_integration.results import (
    SdxlIntegrationResultRecorder,
)
from tools.sdxl_attention_coupling_integration.workflow import (
    build_sdxl_attention_coupling_workflow,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p8.5-sdxl"
)


def execute_sdxl_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    checkpoint_path: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Execute, validate, persist, and clean one complete managed SDXL run."""

    checkpoint = ManagedCheckpointLink(
        source=checkpoint_path,
        source_checkpoint_name=CHECKPOINT_SOURCE_NAME,
        identity=CheckpointArtifactIdentity(
            CHECKPOINT_STABLE_NAME,
            CHECKPOINT_SIZE,
            CHECKPOINT_SHA256,
        ),
    )
    masks = ManagedSdxlSplitMasks(
        input_root=comfy_root / "input",
        run_id=artifacts.run_id,
    )
    recorder = SdxlIntegrationResultRecorder(artifacts.root)
    server_cleanup = False
    system_stats: JsonObject = {}
    with checkpoint:
        with masks:
            workflow = build_sdxl_attention_coupling_workflow(
                run_id=artifacts.run_id,
                checkpoint_name=checkpoint.checkpoint_name,
                mask_names=masks.names,
            )
            mask_evidence = masks.evidence()
            with ManagedComfyServer(
                comfy_root=comfy_root,
                artifacts=artifacts,
                required_node_ids=workflow.required_node_ids,
                readiness_timeout=readiness_timeout,
            ) as running:
                started_at = time.perf_counter()
                prompt_id = running.client.submit(workflow.prompt)
                history = running.client.wait_for_history(
                    prompt_id,
                    timeout=prompt_timeout,
                )
                wall_runtime_ms = (time.perf_counter() - started_at) * 1000.0
                terminal = decode_sdxl_history(history, workflow)
                image_bytes = {
                    mode.mode_id: running.client.download_image(
                        terminal[mode.mode_id].image_reference
                    )
                    for mode in MODES
                }
                system_stats = running.system_stats
                recorder.record_workflow(
                    workflow,
                    history=history,
                    prompt_id=prompt_id,
                    evidence=terminal,
                    image_bytes=image_bytes,
                    masks=mask_evidence,
                    wall_runtime_ms=wall_runtime_ms,
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
        checkpoint_cleanup=checkpoint.cleaned,
        mask_cleanup=masks.cleaned,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command arguments and return one process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=240.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = execute_sdxl_matrix(
            artifacts,
            comfy_root=args.comfy_root,
            checkpoint_path=args.checkpoint_path,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Managed SDXL matrix failed at %s", artifacts.root)
        return 1
    LOGGER.info("Managed SDXL matrix completed: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
