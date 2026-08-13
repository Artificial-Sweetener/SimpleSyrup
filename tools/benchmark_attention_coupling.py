# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute the deterministic Attention Coupling benchmark against ComfyUI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.manifest_decode import json_array, json_object
from tools.attention_coupling_benchmark.manifest_types import (
    BenchmarkManifest,
    JsonObject,
)
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter
from tools.attention_coupling_benchmark.results import (
    BenchmarkResultRecorder,
    parse_completed_outputs,
)
from tools.attention_coupling_benchmark.workflow import BenchmarkWorkflowBuilder
from tools.comfy_api import LoopbackComfyClient

LOGGER = logging.getLogger("simple_syrup.attention_coupling_benchmark")

REQUIRED_NODE_IDS = frozenset(
    {
        "SimpleSyrup.SimpleLoadAnima",
        "SimpleSyrup.EncodePromptBatch",
        "SimpleSyrup.LoadMaskBatch",
        "SimpleSyrup.KSamplerPromptByRegion",
        "SimpleSyrup.KSamplerPromptByTiledRegion",
        "SimpleSyrup.KSamplerContextualDiffusion",
        "SimpleSyrupBenchmark.InstrumentModel",
        "SimpleSyrupBenchmark.ReadMetrics",
        "EmptyCosmosLatentVideo",
        "SolidMask",
        "VAEDecode",
        "SaveImage",
    }
)


def main() -> int:
    """Run or resume the manifest matrix against an already-running server."""

    arguments = _arguments()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = load_manifest()
    client = LoopbackComfyClient(arguments.server_url)
    stats = client.verify_server(REQUIRED_NODE_IDS)
    environment = _environment(manifest, stats)
    recorder = BenchmarkResultRecorder(manifest, arguments.output_root, environment)
    mask_writer = MaskArtifactWriter(
        arguments.comfy_input_root,
        f"simple-syrup-benchmark-{manifest.benchmark_id}",
    )
    workflow_builder = BenchmarkWorkflowBuilder()
    completed = recorder.completed_artifact_ids
    pending = [run for run in manifest.runs() if run.artifact_id not in completed]
    if arguments.max_runs is not None:
        pending = pending[: arguments.max_runs]
    LOGGER.info(
        "Benchmark execution starting",
        extra={"total_runs": len(manifest.runs()), "pending_runs": len(pending)},
    )

    cases = {case.case_id: case for case in manifest.cases}
    for index, run in enumerate(pending, start=1):
        LOGGER.info(
            "Executing benchmark run %s/%s: %s",
            index,
            len(pending),
            run.artifact_id,
        )
        try:
            mask_names = mask_writer.write_case(
                cases[run.case_id],
                width=manifest.sampling.width,
                height=manifest.sampling.height,
            )
            workflow = workflow_builder.build(manifest, run, mask_names)
            prompt_id = client.submit(workflow.prompt)
            history = client.wait_for_history(
                prompt_id,
                timeout=arguments.prompt_timeout,
            )
            outputs = parse_completed_outputs(
                history,
                metrics_node_id=workflow.metrics_node_id,
                save_node_id=workflow.save_node_id,
            )
            image_bytes = client.download_image(outputs.image)
            recorder.record_success(run, outputs, image_bytes)
            LOGGER.info(
                "Benchmark run completed: %s calls, %.2f ms, %.2f GiB peak",
                outputs.metrics.model_call_count,
                outputs.metrics.runtime_ms,
                outputs.metrics.peak_vram_bytes / 1024**3,
            )
        except Exception as error:
            recorder.record_failure(run, error)
            LOGGER.exception("Benchmark run failed: %s", run.artifact_id)
            if arguments.fail_fast:
                return 1

    if arguments.max_runs is not None:
        LOGGER.info(
            "Bounded benchmark smoke batch completed; result remains in-progress."
        )
        return 0
    try:
        result_path = recorder.finalize()
    except ValueError:
        LOGGER.exception("Benchmark matrix did not reach a successful terminal state.")
        return 1
    LOGGER.info("Benchmark result completed: %s", result_path)
    return 0


def _arguments() -> argparse.Namespace:
    """Parse explicit server, input, output, and bounded-execution controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8297")
    parser.add_argument(
        "--comfy-input-root",
        type=Path,
        default=Path(r"<COMFY_ROOT>\input"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p0.6"
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


def _environment(manifest: BenchmarkManifest, stats: JsonObject) -> JsonObject:
    """Build the result-schema environment from server and pinned identities."""

    system = json_object(stats.get("system"), "system_stats.system")
    devices = json_array(stats.get("devices"), "system_stats.devices")
    if not devices:
        raise ValueError("ComfyUI system stats did not report a GPU device.")
    device = json_object(devices[0], "system_stats.devices[0]")
    return {
        "simple_syrup_commit": manifest.simple_syrup_commit,
        "comfyui_commit": manifest.comfyui_commit,
        "python_version": _text(system.get("python_version"), "python_version"),
        "torch_version": _text(system.get("pytorch_version"), "pytorch_version"),
        "cuda_version": "13.0",
        "gpu_name": _text(device.get("name"), "GPU name"),
        "attention_backend": "attention_pytorch",
        "model_hashes": {model.artifact_id: model.sha256 for model in manifest.models},
    }


def _text(value: object, label: str) -> str:
    """Narrow one required system-stats string."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"ComfyUI {label} must be a nonempty string.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
