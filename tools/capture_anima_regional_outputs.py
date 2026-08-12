"""Capture labeled regional Anima LoRA images through an isolated ComfyUI."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from tools.anima_regional_output_capture.matrix import HEIGHT, WIDTH, profiles
from tools.anima_regional_output_capture.results import VisualOutputResultRecorder
from tools.anima_regional_output_capture.workflow import VisualOutputWorkflowBuilder
from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.history_output import extract_saved_image
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer

LOGGER = logging.getLogger(__name__)
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p5.7\visual"
)


def execute_capture(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Run the fixed matrix in one owned server and return its result path."""

    definitions = profiles()
    manifest = load_manifest()
    cases = {case.case_id: case for case in manifest.cases}
    masks = MaskArtifactWriter(
        comfy_root / "input",
        f"p57-visual-{artifacts.run_id.lower()}",
    )
    mask_names_by_case = {
        "vertical-hard-50-50": masks.write_case(
            cases["vertical-hard-50-50"], width=WIDTH, height=HEIGHT
        )
    }
    builder = VisualOutputWorkflowBuilder()
    workflows = tuple(
        builder.build(
            profile,
            run_id=artifacts.run_id,
            mask_names=_mask_names(profile.mask_case_id, mask_names_by_case),
        )
        for profile in definitions
    )
    required = frozenset().union(
        *(workflow.required_node_ids for workflow in workflows)
    )
    recorder = VisualOutputResultRecorder(artifacts.root)
    with ManagedComfyServer(
        comfy_root=comfy_root,
        artifacts=artifacts,
        required_node_ids=required,
        readiness_timeout=readiness_timeout,
    ) as running:
        for profile, workflow in zip(definitions, workflows, strict=True):
            LOGGER.info("Capturing labeled visual profile %s", profile.profile_id)
            prompt_id = running.client.submit(workflow.prompt)
            history = running.client.wait_for_history(prompt_id, timeout=prompt_timeout)
            reference = extract_saved_image(history, workflow.save_node_id)
            image_bytes = running.client.download_image(reference)
            recorder.record(
                profile,
                workflow=workflow.prompt,
                history=history,
                prompt_id=prompt_id,
                image_reference=reference,
                image_bytes=image_bytes,
            )
    artifacts.record_cleanup(
        process_running=running.process.is_running,
        port_available=is_loopback_port_available(running.port),
    )
    return recorder.finalize(definitions)


def _mask_names(
    case_id: str | None,
    names_by_case: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Resolve optional fixed-case mask filenames with explicit narrowing."""

    if case_id is None or case_id == "all-one":
        return ()
    return names_by_case[case_id]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--readiness-timeout", type=float, default=240.0)
    parser.add_argument("--prompt-timeout", type=float, default=1200.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = IntegrationArtifacts(args.output_root)
    try:
        result = execute_capture(
            artifacts,
            comfy_root=args.comfy_root,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("Regional output capture failed at %s", artifacts.root)
        return 1
    LOGGER.info("Regional output capture completed at %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
