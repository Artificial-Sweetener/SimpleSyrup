"""Run the P9.1 Prompt Control matrix in one isolated managed ComfyUI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from tools.anima_attention_coupling_integration.schema import (
    validate_public_node_metadata,
)
from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.prompt_control_attention_coupling_integration.baseline import (
    DEFAULT_BASELINE_PATH,
    load_baseline,
)
from tools.prompt_control_attention_coupling_integration.history import parse_outputs
from tools.prompt_control_attention_coupling_integration.matrix import (
    HEIGHT,
    MASK_CASE_ID,
    PUBLIC_NODE_ID,
    WIDTH,
    cases,
)
from tools.prompt_control_attention_coupling_integration.results import (
    PromptControlAttentionResultRecorder,
)
from tools.prompt_control_attention_coupling_integration.workflow import (
    PromptControlAttentionWorkflowBuilder,
)
from tools.prompt_control_characterization.source_identity import (
    inspect_source,
    validate_pinned_source,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_PROMPT_CONTROL_ROOT = Path(r"<COMFY_ROOT>\custom_nodes\ComfyUI-Prompt-Control")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p9.1"
)


def execute_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    prompt_control_root: Path,
    baseline_path: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Execute all nine cases and return the completed result path."""

    source = inspect_source(prompt_control_root)
    validate_pinned_source(source)
    baseline = load_baseline(baseline_path)
    definitions = cases()
    manifest_case = next(
        case for case in load_manifest().cases if case.case_id == MASK_CASE_ID
    )
    mask_writer = MaskArtifactWriter(
        comfy_root / "input",
        f"p91-{artifacts.run_id.lower()}",
    )
    mask_names = mask_writer.write_case(
        manifest_case,
        width=WIDTH,
        height=HEIGHT,
    )
    recorder = PromptControlAttentionResultRecorder(artifacts.root, baseline, source)
    recorder.record_masks(mask_names, input_root=comfy_root / "input")
    builder = PromptControlAttentionWorkflowBuilder()
    workflows = tuple(
        builder.build(case, run_id=artifacts.run_id, mask_names=mask_names)
        for case in definitions
    )
    required = frozenset().union(
        *(workflow.required_node_ids for workflow in workflows)
    )
    system_stats = {}
    port = 0
    process = None
    try:
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
                LOGGER.info("P9.1 starting %s — %s", case.case_id, case.label)
                started = time.perf_counter()
                prompt_id = running.client.submit(workflow.prompt)
                history = running.client.wait_for_history(
                    prompt_id,
                    timeout=prompt_timeout,
                )
                outputs = parse_outputs(history, workflow)
                reference = extract_saved_image(history, workflow.save_node_id)
                image_bytes = running.client.download_image(reference)
                image_path = recorder.record_case(
                    case,
                    workflow,
                    outputs,
                    history=history,
                    prompt_id=prompt_id,
                    image_reference=reference,
                    image_bytes=image_bytes,
                    wall_runtime_ms=(time.perf_counter() - started) * 1000.0,
                )
                LOGGER.info("P9.1 accepted %s — %s", case.label, image_path)
            system_stats = running.system_stats
            port = running.port
            process = running.process
    finally:
        _remove_owned_masks(comfy_root / "input", mask_names)
    if process is None:
        raise RuntimeError("P9.1 managed Comfy did not reach ready state.")
    process_stopped = not process.is_running
    port_available = is_loopback_port_available(port)
    masks_removed = all(
        not (comfy_root / "input" / name).exists() for name in mask_names
    )
    artifacts.record_cleanup(
        process_running=process.is_running,
        port_available=port_available,
    )
    return recorder.finalize(
        definitions,
        system_stats=system_stats,
        cleanup_verified=process_stopped and port_available and masks_removed,
    )


def _remove_owned_masks(input_root: Path, names: tuple[str, ...]) -> None:
    """Remove only exact uniquely prefixed masks created by this run."""

    resolved_root = input_root.resolve()
    for name in names:
        path = (resolved_root / name).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("P9.1 mask cleanup escaped the Comfy input root.")
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit boundaries and return one process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument(
        "--prompt-control-root",
        type=Path,
        default=DEFAULT_PROMPT_CONTROL_ROOT,
    )
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=300.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
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
            prompt_control_root=args.prompt_control_root,
            baseline_path=args.baseline_path,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("P9.1 integration failed at %s", artifacts.root)
        return 1
    LOGGER.info("P9.1 integration completed at %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
