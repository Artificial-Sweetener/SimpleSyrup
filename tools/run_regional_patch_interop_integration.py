"""Run the closed P9.7 modifier interoperability matrix in managed ComfyUI."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from tools.anima_lora_characterization.artifact_inventory import (
    inspect_adapter,
    validate_pinned_inventory,
)
from tools.anima_lora_characterization.matrix import (
    PINNED_LORA_SHA256,
    PINNED_LORA_SIZE,
)
from tools.attention_coupling_benchmark.manifest import load_manifest
from tools.attention_coupling_benchmark.manifest_types import ModelArtifact
from tools.attention_coupling_benchmark.mask_artifacts import MaskArtifactWriter
from tools.comfy_api import JsonObject
from tools.comfy_integration.artifacts import IntegrationArtifacts
from tools.comfy_integration.loopback_port import is_loopback_port_available
from tools.comfy_integration.managed_server import ManagedComfyServer
from tools.regional_patch_interop_integration.history import (
    RegionalPatchInteropSuccess,
    parse_history,
)
from tools.regional_patch_interop_integration.matrix import (
    FULL_HEIGHT,
    FULL_WIDTH,
    MASK_CASE_ID,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    PatchInteropModelFamily,
    PatchInteropSpatialMode,
    cases,
)
from tools.regional_patch_interop_integration.results import (
    RegionalPatchInteropResultRecorder,
)
from tools.regional_patch_interop_integration.source_identity import (
    inspect_repository_revision,
)
from tools.regional_patch_interop_integration.workflow import (
    CONTEXTUAL_NODE_ID,
    FULL_NODE_ID,
    TILED_NODE_ID,
    RegionalPatchInteropWorkflowBuilder,
)
from tools.sdxl_attention_coupling_integration.checkpoint_link import (
    CheckpointArtifactIdentity,
    ManagedCheckpointLink,
)
from tools.sdxl_attention_coupling_integration.matrix import (
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE,
    CHECKPOINT_SOURCE_NAME,
    CHECKPOINT_STABLE_NAME,
)

LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMFY_ROOT = Path(r"<COMFY_ROOT>")
DEFAULT_LORA_PATH = Path(r"<MODEL_ROOT>\Loras\Anima\style\adapter-a.safetensors")
DEFAULT_SDXL_CHECKPOINT_PATH = (
    Path(r"<MODEL_ROOT>\checkpoints\SDXL")
    / "juggernautXL_juggXIByRundiffusion.safetensors"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"<COMFY_ROOT>\benchmark_artifacts\anima-regional-prompting-v1\p9.7"
)
METADATA_NODE_IDS = (
    FULL_NODE_ID,
    TILED_NODE_ID,
    CONTEXTUAL_NODE_ID,
    "EasyCache",
    "LazyCache",
    "ModelAttentionSelector",
    "CLIPNegPip",
    "SimpleSyrupBenchmark.SnapshotModelModifier",
)


def execute_matrix(
    artifacts: IntegrationArtifacts,
    *,
    comfy_root: Path,
    lora_path: Path,
    sdxl_checkpoint_path: Path,
    readiness_timeout: float,
    prompt_timeout: float,
) -> Path:
    """Execute, validate, persist, and clean the complete ordered matrix."""

    lora_inventory = inspect_adapter(lora_path)
    validate_pinned_inventory(lora_inventory)
    manifest = load_manifest()
    definitions = cases()
    repositories = (
        inspect_repository_revision("SimpleSyrup", REPOSITORY_ROOT),
        inspect_repository_revision("ComfyUI", comfy_root),
    )
    recorder = RegionalPatchInteropResultRecorder(
        artifacts.root,
        repositories=repositories,
        model_inventory=_model_inventory(manifest.models),
    )
    manifest_case = next(
        case for case in manifest.cases if case.case_id == MASK_CASE_ID
    )
    input_root = comfy_root / "input"
    full_masks = MaskArtifactWriter(
        input_root,
        f"p97-{artifacts.run_id.lower()}-full",
    ).write_case(manifest_case, width=FULL_WIDTH, height=FULL_HEIGHT)
    spatial_masks = MaskArtifactWriter(
        input_root,
        f"p97-{artifacts.run_id.lower()}-spatial",
    ).write_case(manifest_case, width=TARGET_WIDTH, height=TARGET_HEIGHT)
    mask_groups = {"full-1024": full_masks, "spatial-1536": spatial_masks}
    recorder.record_masks(mask_groups, input_root=input_root)
    checkpoint = ManagedCheckpointLink(
        source=sdxl_checkpoint_path,
        source_checkpoint_name=CHECKPOINT_SOURCE_NAME,
        identity=CheckpointArtifactIdentity(
            CHECKPOINT_STABLE_NAME,
            CHECKPOINT_SIZE,
            CHECKPOINT_SHA256,
        ),
    )
    builder = RegionalPatchInteropWorkflowBuilder()
    system_stats: JsonObject = {}
    server_cleanup = False
    try:
        with checkpoint:
            workflows = tuple(
                builder.build(
                    case,
                    run_id=artifacts.run_id,
                    mask_names=(
                        full_masks
                        if case.spatial_mode is PatchInteropSpatialMode.FULL
                        else spatial_masks
                    ),
                    sdxl_checkpoint_name=(
                        checkpoint.checkpoint_name
                        if case.model_family is PatchInteropModelFamily.SDXL
                        else None
                    ),
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
                recorder.record_node_metadata(
                    {
                        node_id: running.client.node_metadata(node_id)
                        for node_id in METADATA_NODE_IDS
                    }
                )
                for case, workflow in zip(definitions, workflows, strict=True):
                    LOGGER.info("P9.7 starting %s — %s", case.case_id, case.label)
                    started = time.perf_counter()
                    prompt_id = running.client.submit(workflow.prompt)
                    history = running.client.wait_for_history(
                        prompt_id,
                        timeout=prompt_timeout,
                    )
                    observed = parse_history(history, workflow)
                    image_bytes = None
                    source_image_bytes = None
                    if isinstance(observed, RegionalPatchInteropSuccess):
                        image_bytes = running.client.download_image(
                            observed.image_reference
                        )
                        if observed.source_image_reference is not None:
                            source_image_bytes = running.client.download_image(
                                observed.source_image_reference
                            )
                    artifact_path = recorder.record_case(
                        case,
                        workflow,
                        observed,
                        history=history,
                        prompt_id=prompt_id,
                        wall_runtime_ms=(time.perf_counter() - started) * 1000.0,
                        image_bytes=image_bytes,
                        source_image_bytes=source_image_bytes,
                    )
                    LOGGER.info("P9.7 validated %s — %s", case.label, artifact_path)
                system_stats = running.system_stats
                port = running.port
                process = running.process
            port_available = is_loopback_port_available(port)
            server_cleanup = not process.is_running and port_available
            artifacts.record_cleanup(
                process_running=process.is_running,
                port_available=port_available,
            )
    finally:
        _remove_owned_masks(input_root, (*full_masks, *spatial_masks))
    masks_removed = all(
        not (input_root / name).exists() for name in (*full_masks, *spatial_masks)
    )
    return recorder.finalize(
        definitions,
        system_stats=system_stats,
        cleanup_verified=server_cleanup and masks_removed,
        checkpoint_cleanup_verified=checkpoint.cleaned,
    )


def _model_inventory(models: tuple[ModelArtifact, ...]) -> tuple[JsonObject, ...]:
    """Return pinned Anima, ADAPTER_A, and SDXL identities without local paths."""

    values = tuple(asdict(model) for model in models)
    return (
        *values,
        {
            "artifact_id": "adapter_a-anima",
            "role": "regional_lora",
            "filename": "adapter-a.safetensors",
            "size_bytes": PINNED_LORA_SIZE,
            "sha256": PINNED_LORA_SHA256,
            "target_count": 448,
            "rank": 32,
        },
        {
            "artifact_id": "juggernaut-xl-jugg-xi",
            "role": "sdxl_checkpoint",
            "filename": CHECKPOINT_STABLE_NAME,
            "size_bytes": CHECKPOINT_SIZE,
            "sha256": CHECKPOINT_SHA256,
        },
    )


def _remove_owned_masks(input_root: Path, names: tuple[str, ...]) -> None:
    """Remove only exact uniquely prefixed masks created by this run."""

    resolved_root = input_root.resolve()
    for name in names:
        path = (resolved_root / name).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError("P9.7 mask cleanup escaped the Comfy input root.")
        path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse explicit boundaries and return one process exit status."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument(
        "--sdxl-checkpoint-path",
        type=Path,
        default=DEFAULT_SDXL_CHECKPOINT_PATH,
    )
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
            lora_path=args.lora_path,
            sdxl_checkpoint_path=args.sdxl_checkpoint_path,
            readiness_timeout=args.readiness_timeout,
            prompt_timeout=args.prompt_timeout,
        )
    except BaseException as error:
        artifacts.record_failure(error)
        LOGGER.exception("P9.7 integration failed at %s", artifacts.root)
        return 1
    LOGGER.info("P9.7 integration completed at %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
