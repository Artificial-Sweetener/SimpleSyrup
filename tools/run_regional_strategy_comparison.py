"""Run the P10.2 regional strategy comparison in an isolated ComfyUI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.regional_strategy_comparison.history import parse_completed_evidence
from tools.regional_strategy_comparison.input_artifacts import (
    ComparisonInputArtifacts,
)
from tools.regional_strategy_comparison.matrix import (
    MASK_CASE_ID,
    SOURCE_CASE_ID,
    StrategyComparisonCase,
    cases,
)
from tools.regional_strategy_comparison.results import (
    StrategyComparisonResultRecorder,
)
from tools.regional_strategy_comparison.workflow import (
    BuiltStrategyComparisonWorkflow,
    StrategyComparisonWorkflowBuilder,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p10.2"
)


def execute_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Execute the complete comparison and return the accepted result path."""

    definitions = cases()
    manifest_case = next(
        case for case in load_manifest().cases if case.case_id == MASK_CASE_ID
    )
    recorder = StrategyComparisonResultRecorder(artifacts.root)
    inputs = ComparisonInputArtifacts(
        comfy_root / "input",
        artifacts.root,
        artifacts.run_id,
    )
    with inputs:
        masks = inputs.materialize_masks(manifest_case)
        workflows = _workflows(
            definitions,
            run_id=artifacts.run_id,
            full_masks=masks.full,
            refinement_masks=masks.refinement,
            shared_source_name=inputs.source_name,
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
            metadata = _public_node_metadata(running.client.node_metadata, workflows)
            recorder.record_metadata(metadata)
            for case, workflow in zip(definitions, workflows, strict=True):
                if (
                    case.is_refinement
                    and not (comfy_root / "input" / inputs.source_name).is_file()
                ):
                    raise RuntimeError(
                        "P10.2 refinement started before source publication."
                    )
                LOGGER.info("P10.2 starting %s — %s", case.case_id, case.label)
                started = time.perf_counter()
                prompt_id = running.client.submit(workflow.prompt)
                history = running.client.wait_for_history(
                    prompt_id, timeout=prompt_timeout
                )
                recorder.record_raw_case(case, workflow, history)
                evidence = parse_completed_evidence(history, workflow)
                image_bytes = running.client.download_image(evidence.image_reference)
                image_path = recorder.record_case(
                    case,
                    workflow,
                    evidence,
                    history=history,
                    prompt_id=prompt_id,
                    image_bytes=image_bytes,
                    wall_runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
                if case.case_id == SOURCE_CASE_ID:
                    inputs.publish_source(image_bytes)
                LOGGER.info("P10.2 accepted case artifact %s", image_path)
            system_stats = running.system_stats
            port = running.port
            process = running.process
    process_cleanup = not process.is_running and is_loopback_port_available(port)
    cleanup_verified = process_cleanup and inputs.cleanup_verified
    artifacts.record_cleanup(
        process_running=process.is_running,
        port_available=is_loopback_port_available(port),
    )
    return recorder.finalize(
        definitions,
        source_sha256=inputs.source_sha256,
        system_stats=system_stats,
        cleanup_verified=cleanup_verified,
    )


def _workflows(
    definitions: tuple[StrategyComparisonCase, ...],
    *,
    run_id: str,
    full_masks: tuple[str, ...],
    refinement_masks: tuple[str, ...],
    shared_source_name: str,
) -> tuple[BuiltStrategyComparisonWorkflow, ...]:
    """Build the complete ordered graph matrix from one workflow owner."""

    builder = StrategyComparisonWorkflowBuilder()
    return tuple(
        builder.build(
            case,
            run_id=run_id,
            mask_names=refinement_masks if case.is_refinement else full_masks,
            shared_source_name=shared_source_name,
        )
        for case in definitions
    )


def _public_node_metadata(
    read_metadata: Callable[[str], JsonObject],
    workflows: tuple[BuiltStrategyComparisonWorkflow, ...],
) -> JsonObject:
    """Read each unique compared public sampler's live metadata."""

    node_ids = sorted(
        {
            str(node["class_type"])
            for workflow in workflows
            for node in workflow.prompt.values()
            if str(node["class_type"]).startswith("SimpleSyrup.KSampler")
        }
    )
    return {node_id: read_metadata(node_id) for node_id in node_ids}


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit status."""

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
        LOGGER.exception("P10.2 comparison failed at %s", artifacts.root)
        return 1
    LOGGER.info("P10.2 comparison completed at %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
