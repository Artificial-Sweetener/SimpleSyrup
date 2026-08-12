"""Capture pinned Prompt Control schedules through an existing ComfyUI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tools.comfy_api import JsonObject, LoopbackComfyClient
from tools.prompt_control_characterization.cases import PromptControlCase, cases
from tools.prompt_control_characterization.history_outputs import parse_outputs
from tools.prompt_control_characterization.results import PromptControlResultRecorder
from tools.prompt_control_characterization.source_identity import (
    PromptControlSourceIdentity,
    inspect_source,
    validate_pinned_source,
)
from tools.prompt_control_characterization.workflow import (
    PromptControlWorkflowBuilder,
)

LOGGER = logging.getLogger("simple_syrup.prompt_control_characterization")


def main() -> int:
    """Run or resume every P0.8 case against one loopback Comfy server."""

    arguments = _arguments()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    source = inspect_source(arguments.prompt_control_root)
    validate_pinned_source(source)
    matrix = cases()
    builder = PromptControlWorkflowBuilder()
    client = LoopbackComfyClient(arguments.server_url)
    workflows = {case.case_id: builder.build(case) for case in matrix}
    required_nodes = {
        _text(node.get("class_type"), "workflow node class")
        for workflow in workflows.values()
        for node in workflow.prompt.values()
    }
    stats = client.verify_server(required_nodes)
    recorder = PromptControlResultRecorder(
        arguments.output_root,
        source,
        _environment(stats, source),
        matrix,
    )
    pending = [
        case for case in matrix if case.case_id not in recorder.completed_case_ids
    ]
    for index, case in enumerate(pending, start=1):
        if not _execute(
            case,
            index=index,
            total=len(pending),
            workflow=workflows[case.case_id],
            client=client,
            recorder=recorder,
            timeout=arguments.prompt_timeout,
        ):
            return 1
    try:
        result = recorder.finalize()
    except ValueError:
        LOGGER.exception(
            "Prompt Control matrix did not reach a successful terminal state."
        )
        return 1
    LOGGER.info("Prompt Control characterization completed: %s", result)
    return 0


def _execute(
    case: PromptControlCase,
    *,
    index: int,
    total: int,
    workflow: object,
    client: LoopbackComfyClient,
    recorder: PromptControlResultRecorder,
    timeout: float,
) -> bool:
    """Execute and record one case without owning server lifecycle."""

    from tools.prompt_control_characterization.workflow import (
        BuiltPromptControlWorkflow,
    )

    if not isinstance(workflow, BuiltPromptControlWorkflow):
        raise TypeError("Prompt Control workflow must be a built workflow.")
    LOGGER.info("Executing Prompt Control case %s/%s: %s", index, total, case.case_id)
    try:
        prompt_id = client.submit(workflow.prompt)
        history = client.wait_for_history(prompt_id, timeout=timeout)
        outputs = parse_outputs(
            history,
            expansion_node_id=workflow.expansion_node_id,
            snapshot_node_id=workflow.snapshot_node_id,
            runtime_node_id=workflow.runtime_node_id,
        )
        recorder.record_success(case, outputs, workflow.prompt)
        LOGGER.info("Prompt Control case completed: %s", case.case_id)
        return True
    except Exception as error:
        recorder.record_failure(case, error)
        LOGGER.exception("Prompt Control case failed: %s", case.case_id)
        return False


def _environment(stats: JsonObject, source: PromptControlSourceIdentity) -> JsonObject:
    """Record stable runtime identity without local install paths."""

    system = stats.get("system")
    devices = stats.get("devices")
    if not isinstance(system, dict) or not isinstance(devices, list) or not devices:
        raise TypeError("Comfy system stats must contain system and device data.")
    device = devices[0]
    if not isinstance(device, dict):
        raise TypeError("Comfy primary device stats must be an object.")
    return {
        "comfyui_commit": "a1c421994cdcc5044dbce2bb7628e89386311cc5",
        "prompt_control_version": source.version,
        "python_version": _text(system.get("python_version"), "python_version"),
        "torch_version": _text(system.get("pytorch_version"), "torch_version"),
        "gpu_name": _text(device.get("name"), "gpu_name"),
        "server_url": "http://127.0.0.1:8297",
    }


def _arguments() -> argparse.Namespace:
    """Parse explicit install, output, and loopback server locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-control-root",
        type=Path,
        default=Path(r"<COMFY_ROOT>\custom_nodes\ComfyUI-Prompt-Control"),
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8297")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p0.8"
        ),
    )
    parser.add_argument("--prompt-timeout", type=float, default=600.0)
    return parser.parse_args()


def _text(value: object, field: str) -> str:
    """Narrow one required string."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be nonempty text.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
