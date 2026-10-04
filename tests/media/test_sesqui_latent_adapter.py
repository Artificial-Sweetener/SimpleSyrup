"""Exercise actual learned inference in Comfy's external latent units and layouts."""

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from simple_syrup.runtime.sesqui_latent_upscale import upscale_with_sesqui


@pytest.mark.parametrize(
    "family,channels,video",
    (("sdxl", 4, False), ("flux", 16, False), ("wan", 16, True), ("flux2", 128, False)),
)
def test_learned_adapter_matches_direct_raw_vae_inference(
    tmp_path: Path, family: str, channels: int, video: bool
) -> None:
    """Detect extra latent normalization, packed Flux2 mistakes and frame transposes."""
    vendor: Any = import_module("simple_syrup.third_party.sesqui_lsr")
    raw_channels = 32 if channels == 128 else channels
    with torch.random.fork_rng():
        torch.manual_seed(42)
        network = vendor.LatentUpscaler(in_channels=raw_channels).eval()
        samples = torch.randn(
            (2, channels, 3, 8, 12) if video else (2, channels, 8, 12)
        )
    path = tmp_path / "learned.safetensors"
    save_file(network.state_dict(), str(path))
    flat = (
        samples.permute(0, 2, 1, 3, 4).reshape(6, channels, 8, 12) if video else samples
    )
    adaptor = vendor.make_flux2() if channels == 128 else vendor.make_identity(channels)
    with torch.inference_mode():
        expected = adaptor.from_vae_latent(
            network(adaptor.to_vae_latent(flat), adaptor.vae_target_size((12, 18)))
        )
    if video:
        expected = expected.reshape(2, 3, channels, 12, 18).permute(0, 2, 1, 3, 4)
    output = upscale_with_sesqui(samples, path=path, family=family, target=(12, 18))
    assert output.shape == expected.shape and output.dtype == samples.dtype
    assert torch.equal(output, expected)


def test_invalid_channels_fail_before_checkpoint_loading(tmp_path: Path) -> None:
    """Reject incompatible learned format layouts before model IO occurs."""
    with pytest.raises(ValueError, match="16 channels"):
        upscale_with_sesqui(
            torch.zeros(1, 4, 8, 12),
            path=tmp_path / "missing.safetensors",
            family="wan",
            target=(12, 18),
        )
