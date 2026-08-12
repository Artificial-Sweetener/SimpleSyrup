"""Build minimal complete P0.7 evidence for focused owner tests."""

from __future__ import annotations

from tools.anima_lora_characterization.artifact_inventory import (
    AdapterInventory,
    AdapterPair,
)
from tools.anima_lora_characterization.history_outputs import LoraCompletedOutputs
from tools.anima_lora_characterization.matrix import LoraRun
from tools.comfy_api import ImageReference, JsonObject

TARGET = "diffusion_model.blocks.0.cross_attn.q_proj"
TARGET_KEY = f"{TARGET}.weight"


def inventory() -> AdapterInventory:
    """Return one valid synthetic Anima target inventory."""

    return AdapterInventory(
        size_bytes=10,
        sha256="a" * 64,
        metadata={"modelspec.architecture": "anima/lora"},
        pairs=(AdapterPair(TARGET, 0, "cross_attn.q_proj", 4, 8, 16),),
    )


def completed_outputs(run: LoraRun) -> LoraCompletedOutputs:
    """Return complete evidence matching one fixed run profile."""

    return LoraCompletedOutputs(
        metrics=metrics(run),
        image=ImageReference("image.png", "benchmark", "output"),
    )


def metrics(run: LoraRun) -> JsonObject:
    """Build exact static or scheduled probe evidence for one run."""

    adapters = run.profile.adapters
    static_order = [
        {
            "identity": adapter.identity,
            "order": index,
            "strength_patch": adapter.strength,
            "patch_type": "LoRAAdapter",
            "strength_model": 1.0,
        }
        for index, adapter in enumerate(adapters)
    ]
    is_static = run.profile.mode == "static"
    static_active = is_static and bool(adapters)
    hooks = []
    schedules = []
    if not is_static:
        hooks = [
            {
                "identity": adapter.identity,
                "order": index,
                "base_strength_model": adapter.strength,
                "schedule_strength": 0.0,
                "effective_strength_model": 0.0,
                "target_count": 1,
                "target_keys": [TARGET_KEY],
            }
            for index, adapter in enumerate(adapters)
        ]
        schedules = [
            {
                "call_index": call_index,
                "timestep": timestep,
                "effective_strengths": [
                    adapter.strength * multiplier for adapter in adapters
                ],
            }
            for call_index, timestep, multiplier in (
                (1, 1.0, 0.0),
                (9, 0.9, 1.0),
                (16, 0.75, 0.5),
                (24, 0.48, 0.0),
            )
        ]
    denoiser = (
        [
            {
                "call_index": index,
                "shape": [1, 2, 2, 2],
                "source_dtype": "torch.float16",
                "float32_sha256": f"{index:064x}",
            }
            for index in range(1, 31)
        ]
        if run.capture_outputs
        else []
    )
    return {
        "run_id": run.artifact_id,
        "capture_outputs": run.capture_outputs,
        "runtime_ms": 10.0,
        "model_call_count": 30,
        "peak_vram_bytes": 100,
        "static_patches": {
            "target_count": 1 if static_active else 0,
            "target_keys": [TARGET_KEY] if static_active else [],
            "entry_count_histogram": ({str(len(adapters)): 1} if static_active else {}),
            "canonical_target": TARGET_KEY if static_active else None,
            "canonical_patch_order": static_order if is_static else [],
            "inconsistent_order_targets": [],
        },
        "hook_patches": hooks,
        "schedule_observations": schedules,
        "denoiser_outputs": denoiser,
    }
