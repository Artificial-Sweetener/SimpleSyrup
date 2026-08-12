"""Run the P7.8 combined public-node matrix in isolated ComfyUI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from tools.anima_contextual_attention_coupling_integration.matrix import (
    PUBLIC_NODE_ID,
    SUBTOKEN_MASK_CASE_ID,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    cases,
    select_cases,
    subtoken_mask_case,
)
from tools.anima_contextual_attention_coupling_integration.results import (
    ContextualIntegrationResultRecorder,
)
from tools.anima_contextual_attention_coupling_integration.schema import (
    validate_public_node_metadata,
)
from tools.anima_contextual_attention_coupling_integration.workflow import (
    ContextualIntegrationWorkflowBuilder,
)
from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p7.8"
)


def execute_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
    case_ids: tuple[str, ...] = (),
) -> Path:
    """Execute every selected Contextual case and return the result path."""

    definitions = select_cases(case_ids)
    manifest_cases = {case.case_id: case for case in load_manifest().cases}
    manifest_cases[SUBTOKEN_MASK_CASE_ID] = subtoken_mask_case()
    mask_writer = MaskArtifactWriter(
        comfy_root / "input",
        f"p78-{artifacts.run_id.lower()}",
    )
    required_mask_cases = {case.mask_case_id for case in definitions}
    names_by_case = {
        case_id: mask_writer.write_case(
            manifest_cases[case_id],
            width=TARGET_WIDTH,
            height=TARGET_HEIGHT,
        )
        for case_id in sorted(required_mask_cases)
    }
    recorder = ContextualIntegrationResultRecorder(artifacts.root)
    recorder.record_masks(names_by_case, input_root=comfy_root / "input")
    builder = ContextualIntegrationWorkflowBuilder()
    workflows = tuple(
        builder.build(
            case,
            run_id=artifacts.run_id,
            mask_names=names_by_case[case.mask_case_id],
        )
        for case in definitions
    )
    required = frozenset().union(
        *(workflow.required_node_ids for workflow in workflows)
    )
    with ManagedComfyServer(
        comfy_root=comfy_root,
        artifacts=artifacts,
        required_node_ids=required,
        readiness_timeout=readiness_timeout,
    ) as running:
        metadata = running.client.node_metadata(PUBLIC_NODE_ID)
        validate_public_node_metadata(metadata)
        recorder.record_metadata(metadata)
        for case, workflow in zip(definitions, workflows, strict=True):
            LOGGER.info("P7.8 starting %s — %s", case.case_id, case.label)
            started = time.perf_counter()
            prompt_id = running.client.submit(workflow.prompt)
            history = running.client.wait_for_history(prompt_id, timeout=prompt_timeout)
            reference = extract_saved_image(history, workflow.save_node_id)
            image_bytes = running.client.download_image(reference)
            if workflow.source_save_node_id is None:
                raise RuntimeError(
                    "P7.8 product workflow is missing its coherent source image."
                )
            source_reference = extract_saved_image(
                history,
                workflow.source_save_node_id,
            )
            source_image_bytes = running.client.download_image(source_reference)
            image_path = recorder.record_case(
                case,
                workflow,
                history=history,
                prompt_id=prompt_id,
                image_reference=reference,
                image_bytes=image_bytes,
                source_image_reference=source_reference,
                source_image_bytes=source_image_bytes,
                wall_runtime_ms=(time.perf_counter() - started) * 1000.0,
            )
            LOGGER.info("P7.8 artifact %s — %s", case.label, image_path)
        system_stats = running.system_stats
        port = running.port
        process = running.process
    cleanup_verified = not process.is_running and is_loopback_port_available(port)
    artifacts.record_cleanup(
        process_running=process.is_running,
        port_available=is_loopback_port_available(port),
    )
    return recorder.finalize(
        definitions,
        system_stats=system_stats,
        cleanup_verified=cleanup_verified,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--prompt-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.case_id for case in cases()),
        default=[],
        help="Run one case; repeat for an ordered diagnostic subset.",
    )
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
            case_ids=tuple(args.case),
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("P7.8 Contextual integration failed at %s", artifacts.root)
        return 1
    LOGGER.info("P7.8 Contextual integration completed at %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
