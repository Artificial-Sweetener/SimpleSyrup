"""Capture the fixed global ADAPTER_A LoRA matrix through an existing ComfyUI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tools.anima_lora_characterization.artifact_inventory import (
    AdapterInventory,
    inspect_adapter,
    validate_pinned_inventory,
)
from tools.anima_lora_characterization.history_outputs import (
    parse_completed_outputs,
)
from tools.anima_lora_characterization.json_contract import (
    integer_value,
    number_value,
)
from tools.anima_lora_characterization.matrix import LoraRun, runs
from tools.anima_lora_characterization.results import LoraResultRecorder
from tools.anima_lora_characterization.workflow import LoraWorkflowBuilder
from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.manifest_types import BenchmarkManifest
from tools.comfy_api import JsonObject, LoopbackComfyClient

LOGGER = logging.getLogger("simple_syrup.anima_lora_characterization")


def main() -> int:
    """Run or resume the complete pinned matrix against one loopback server."""

    arguments = _arguments()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = load_manifest()
    inventory = inspect_adapter(arguments.lora_path)
    validate_pinned_inventory(inventory)
    client = LoopbackComfyClient(arguments.server_url)
    workflow_builder = LoraWorkflowBuilder()
    run_matrix = runs()
    positive_prompt = _positive_prompt(manifest)
    required_nodes = {
        _text(node.get("class_type"), "workflow node class")
        for run in run_matrix
        for node in workflow_builder.build(
            run, manifest.sampling, positive_prompt
        ).prompt.values()
    }
    stats = client.verify_server(required_nodes)
    recorder = LoraResultRecorder(
        arguments.output_root,
        inventory,
        _environment(manifest, inventory, stats),
    )
    pending = [
        run
        for run in run_matrix
        if run.artifact_id not in recorder.completed_artifact_ids
    ]
    if arguments.max_runs is not None:
        pending = pending[: arguments.max_runs]
    LOGGER.info(
        "Pinned ADAPTER_A characterization starting: total=%s pending=%s",
        len(run_matrix),
        len(pending),
    )
    for index, run in enumerate(pending, start=1):
        if (
            not _execute_run(
                run,
                index=index,
                total=len(pending),
                manifest=manifest,
                positive_prompt=positive_prompt,
                workflow_builder=workflow_builder,
                client=client,
                recorder=recorder,
                prompt_timeout=arguments.prompt_timeout,
            )
            and arguments.fail_fast
        ):
            return 1
    if arguments.max_runs is not None:
        LOGGER.info("Bounded ADAPTER_A smoke batch completed; result remains in-progress.")
        return 0
    try:
        result_path = recorder.finalize()
    except ValueError:
        LOGGER.exception(
            "Pinned ADAPTER_A matrix did not reach a successful terminal state."
        )
        return 1
    LOGGER.info("Pinned ADAPTER_A result completed: %s", result_path)
    return 0


def _execute_run(
    run: LoraRun,
    *,
    index: int,
    total: int,
    manifest: BenchmarkManifest,
    positive_prompt: str,
    workflow_builder: LoraWorkflowBuilder,
    client: LoopbackComfyClient,
    recorder: LoraResultRecorder,
    prompt_timeout: float,
) -> bool:
    """Execute and durably record one run without owning retry policy."""

    LOGGER.info("Executing ADAPTER_A run %s/%s: %s", index, total, run.artifact_id)
    try:
        workflow = workflow_builder.build(run, manifest.sampling, positive_prompt)
        prompt_id = client.submit(workflow.prompt)
        history = client.wait_for_history(prompt_id, timeout=prompt_timeout)
        outputs = parse_completed_outputs(
            history,
            metrics_node_id=workflow.metrics_node_id,
            save_node_id=workflow.save_node_id,
        )
        image_bytes = client.download_image(outputs.image)
        recorder.record_success(run, outputs, image_bytes)
        model_call_count = integer_value(
            outputs.metrics.get("model_call_count"), "model_call_count"
        )
        runtime_ms = number_value(outputs.metrics.get("runtime_ms"), "runtime_ms")
        peak_vram_bytes = integer_value(
            outputs.metrics.get("peak_vram_bytes"), "peak_vram_bytes"
        )
        LOGGER.info(
            "ADAPTER_A run completed: calls=%s runtime_ms=%.2f peak_GiB=%.2f",
            model_call_count,
            runtime_ms,
            peak_vram_bytes / 1024**3,
        )
        return True
    except Exception as error:
        recorder.record_failure(run, error)
        LOGGER.exception("ADAPTER_A run failed: %s", run.artifact_id)
        return False


def _positive_prompt(manifest: BenchmarkManifest) -> str:
    """Use the fixed global-only P0.5 prompt without regional semantics."""

    candidates = [
        case.global_prompt for case in manifest.cases if not case.regional_prompts
    ]
    if len(candidates) != 1:
        raise ValueError("The benchmark manifest must contain one global-only case.")
    return candidates[0]


def _environment(
    manifest: BenchmarkManifest,
    inventory: AdapterInventory,
    stats: JsonObject,
) -> JsonObject:
    """Record stable runtime and model identities without private paths."""

    system = _object(stats.get("system"), "system_stats.system")
    devices = _array(stats.get("devices"), "system_stats.devices")
    if not devices:
        raise ValueError("ComfyUI system stats did not report a GPU device.")
    device = _object(devices[0], "system_stats.devices[0]")
    return {
        "simple_syrup_commit": manifest.simple_syrup_commit,
        "comfyui_commit": manifest.comfyui_commit,
        "python_version": _text(system.get("python_version"), "python_version"),
        "torch_version": _text(system.get("pytorch_version"), "pytorch_version"),
        "gpu_name": _text(device.get("name"), "GPU name"),
        "attention_backend": "attention_pytorch",
        "adapter_architecture": inventory.metadata.get("modelspec.architecture", ""),
        "adapter_training_software": inventory.metadata.get("software", ""),
        "model_hashes": {model.artifact_id: model.sha256 for model in manifest.models},
    }


def _arguments() -> argparse.Namespace:
    """Parse explicit private input and external output locations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8297")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p0.7"
        ),
    )
    parser.add_argument("--prompt-timeout", type=float, default=1800.0)
    parser.add_argument("--max-runs", type=_positive_integer)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _positive_integer(value: str) -> int:
    """Parse one positive CLI integer."""

    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _object(value: object, label: str) -> JsonObject:
    """Narrow a system-stats JSON object."""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"ComfyUI {label} must be a JSON object.")
    return value


def _array(value: object, label: str) -> list[object]:
    """Narrow a system-stats JSON array."""

    if not isinstance(value, list):
        raise ValueError(f"ComfyUI {label} must be a JSON array.")
    return value


def _text(value: object, label: str) -> str:
    """Narrow one required system-stats string."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"ComfyUI {label} must be a nonempty string.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
