"""Adapt SesquiLSR inference to Comfy external latent layouts and device management."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from ..domain.latent_upscale import learned_upscale_stages
from ..shared.logging import get_logger

LOGGER = get_logger(__name__)
_CHANNELS = {"sdxl": 4, "flux": 16, "wan": 16, "flux2": 32}


def upscale_with_sesqui(
    samples: torch.Tensor, *, path: Path, family: str, target: tuple[int, int]
) -> torch.Tensor:
    """Run float32 learned stages in raw VAE latent units, without image decoding."""
    module: Any = import_module("simple_syrup.third_party.sesqui_lsr")
    management = import_module("comfy.model_management")
    utils = import_module("comfy.utils")
    channels = _CHANNELS[family]
    external_channels = (
        128 if family == "flux2" and samples.shape[1] == 128 else channels
    )
    if samples.ndim not in (4, 5) or samples.shape[1] != external_channels:
        raise ValueError(
            f"SesquiLSR {family} requires BCHW or BCTHW latents with "
            f"{external_channels} channels."
        )
    if not samples.is_floating_point() or not bool(torch.isfinite(samples).all()):
        raise ValueError("Latent upscaling requires finite floating-point samples.")
    network = module.LatentUpscaler(in_channels=channels)
    network.load_state_dict(
        utils.load_torch_file(str(path), safe_load=True), strict=True
    )
    device = management.get_torch_device()
    management.free_memory(
        management.module_size(network) + samples.numel() * 64, device
    )
    network.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    batch, frames = (
        int(samples.shape[0]),
        int(samples.shape[2]) if samples.ndim == 5 else 1,
    )
    flat = (
        samples.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, external_channels, *samples.shape[-2:]
        )
        if samples.ndim == 5
        else samples
    )
    adaptor = (
        module.make_flux2()
        if external_channels == 128
        else module.make_identity(channels)
    )
    current: torch.Tensor = adaptor.to_vae_latent(
        flat.to(device=device, dtype=torch.float32)
    )
    vae_target = adaptor.vae_target_size(target)
    stages = learned_upscale_stages(
        (int(current.shape[-2]), int(current.shape[-1])), vae_target
    )
    try:
        with torch.inference_mode():
            for size in stages:
                current = network(current, size)
        output: torch.Tensor = adaptor.from_vae_latent(current.float()).to(samples)
    finally:
        network.to(management.unet_offload_device())
    if output.shape[-2:] != target:
        raise RuntimeError(
            "Learned latent upscaler did not produce the requested dimensions."
        )
    if samples.ndim == 5:
        output = output.reshape(batch, frames, external_channels, *target).permute(
            0, 2, 1, 3, 4
        )
    LOGGER.info(
        "Learned latent upscale completed",
        extra={
            "operation": "latent_upscale",
            "family": family,
            "model": path.name,
            "stages": stages,
            "output_shape": tuple(output.shape),
        },
    )
    return output
