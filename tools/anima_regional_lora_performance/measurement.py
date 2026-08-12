"""Measure one exact CUDA denoiser-call sequence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter

import comfy.model_management
import torch

from .runtime_profile import PerformanceRuntimeProfile


@dataclass(frozen=True, slots=True)
class PerformanceSequenceMeasurement:
    """Retain timing, memory, final tensor, identity, and call-count evidence."""

    runtime_ms: float
    peak_vram_bytes: int
    output_sha256: str
    denoiser_calls: int
    output: torch.Tensor


def activate_measurement_model(model: object) -> None:
    """Fully load one source or derived MODEL outside the measured interval."""

    comfy.model_management.load_models_gpu([model], force_full_load=True)


def execute_call_sequence(
    profile: PerformanceRuntimeProfile,
    *,
    latent: torch.Tensor,
    context: torch.Tensor,
    sample_sigmas: torch.Tensor,
    call_count: int,
    measured: bool,
) -> PerformanceSequenceMeasurement:
    """Execute one exact prefix or complete sequence and return evidence."""

    transformer_options = profile.model_options["transformer_options"]
    transformer_options["cond_or_uncond"] = [0, 1]
    transformer_options["sample_sigmas"] = sample_sigmas
    output: torch.Tensor | None = None
    if measured:
        torch.cuda.reset_peak_memory_stats(latent.device)
    torch.cuda.synchronize(latent.device)
    started = perf_counter()
    with torch.inference_mode():
        for sigma in sample_sigmas[:call_count]:
            current = sigma.expand(int(latent.shape[0]))
            transformer_options["sigmas"] = current
            value = profile.model.model.apply_model(
                latent,
                current,
                c_crossattn=context,
                transformer_options=transformer_options,
            )
            if not isinstance(value, torch.Tensor) or value.shape != latent.shape:
                raise TypeError(
                    "Anima performance denoiser output has an invalid tensor shape."
                )
            output = value
    torch.cuda.synchronize(latent.device)
    if output is None:
        raise AssertionError("Anima performance execution produced no output.")
    return PerformanceSequenceMeasurement(
        (perf_counter() - started) * 1000.0,
        torch.cuda.max_memory_allocated(latent.device),
        _tensor_sha256(output),
        call_count,
        output.detach(),
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash one observed output through canonical float32 CPU bytes."""

    value = tensor.detach().float().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
