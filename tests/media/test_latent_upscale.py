# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify native geometry, capture preservation and learned format selection."""

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import torch

from simple_syrup.domain.latent_upscale import (
    learned_upscale_stages,
    recommended_family,
)
from simple_syrup.domain.noise_reuse import PROVENANCE_KEY, SamplingProvenance
from simple_syrup.runtime import latent_upscale_models
from simple_syrup.runtime.latent_upscale_catalog import latent_model_choices
from simple_syrup.runtime.latent_upscale_models import resolve_latent_upscale_model
from simple_syrup.services.latent_upscale_service import LatentUpscaleService


def test_stages_reach_three_times_without_exceeding_trained_range() -> None:
    """Use 2x then 1.5x and preserve exact rounded endpoint dimensions."""
    assert learned_upscale_stages((96, 168), (288, 504)) == ((192, 336), (288, 504))
    assert learned_upscale_stages((192, 256), (288, 384)) == ((288, 384),)
    assert learned_upscale_stages((4, 4), (4, 4)) == ()
    with pytest.raises(ValueError, match="enlargement"):
        learned_upscale_stages((8, 8), (4, 4))


def test_native_comfy_resize_preserves_provenance_and_resizes_mask() -> None:
    """Exercise the real Comfy native node and preserve capture ownership."""
    samples = torch.arange(96).float().reshape(1, 4, 4, 6)
    provenance = SamplingProvenance("SDXL", samples)
    latent: dict[str, Any] = {
        "samples": samples,
        PROVENANCE_KEY: provenance,
        "noise_mask": torch.ones(1, 4, 6),
    }
    output = LatentUpscaleService().upscale(latent, "nearest-exact", 1.5)
    assert output["samples"].shape == (1, 4, 6, 9)
    assert output["noise_mask"].shape == (1, 6, 9)
    assert output[PROVENANCE_KEY] is provenance
    assert latent["samples"] is samples and latent["noise_mask"].shape == (1, 4, 6)


@pytest.mark.parametrize(
    "latent_format,family",
    (
        ("SDXL", "sdxl"),
        ("Flux", "flux"),
        ("Flux2", "flux2"),
        ("Wan21", "wan"),
        ("Cosmos1CV8x8x8", "wan"),
    ),
)
def test_auto_uses_verified_format_instead_of_channel_guess(
    latent_format: str, family: str
) -> None:
    """Distinguish Flux and Wan even though both can have 16 channels."""
    assert recommended_family(latent_format) == family


def test_auto_and_wrong_family_fail_before_download() -> None:
    """Reject ambiguous source identity and incompatible explicit checkpoints."""
    with pytest.raises(ValueError, match="provenance"):
        resolve_latent_upscale_model({"samples": torch.zeros(1, 16, 4, 4)}, "Auto")
    with pytest.raises(ValueError, match="does not match"):
        resolve_latent_upscale_model(
            {PROVENANCE_KEY: SamplingProvenance("Flux", torch.zeros(1, 16, 4, 4))},
            "wan/upscaler_Wan21.safetensors",
        )
    with pytest.raises(ValueError, match="compatible"):
        recommended_family("SD15")


def test_method_choices_deduplicate_registered_windows_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one choice per installed checkpoint while retaining native algorithms."""
    folder_paths = import_module("folder_paths")

    monkeypatch.setattr(
        folder_paths,
        "get_filename_list",
        lambda category: [
            r"wan\upscaler_Wan21.safetensors",
            r"custom\example.safetensors",
        ],
    )
    choices = latent_model_choices()
    assert choices[0] == "Auto"
    assert choices.count("wan/upscaler_Wan21.safetensors") == 1
    assert "custom/example.safetensors" in choices and "nearest-exact" in choices


def test_invalid_samples_and_masks_fail_before_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid downloads or inference when tensor inputs are invalid."""
    from simple_syrup.services import latent_upscale_service

    def forbidden(*args: Any) -> Any:
        """Detect side effects before input validation completes."""
        raise AssertionError("Model resolution must not occur.")

    monkeypatch.setattr(
        latent_upscale_service, "resolve_latent_upscale_model", forbidden
    )
    service = LatentUpscaleService()
    with pytest.raises(ValueError, match="finite"):
        service.upscale({"samples": torch.full((1, 4, 8, 12), float("nan"))}, "Auto", 2)
    with pytest.raises(TypeError, match="mask"):
        service.upscale(
            {"samples": torch.zeros(1, 4, 8, 12), "noise_mask": "invalid"}, "Auto", 2
        )


def test_registered_original_filename_retains_family_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Distinguish same-channel Flux and Wan weights outside the default folders."""
    folders = import_module("folder_paths")
    utils = import_module("comfy.utils")
    monkeypatch.setattr(
        folders,
        "get_full_path_or_raise",
        lambda category, name: str(
            tmp_path / "custom_models" / "upscaler_Flux.safetensors"
        ),
    )
    monkeypatch.setattr(
        utils,
        "load_torch_file",
        lambda path, **kwargs: {
            "head_conv.0.weight": torch.zeros(8, 16, 3, 3),
            "upsample.base_taps": torch.ones(1),
        },
    )
    latent = {PROVENANCE_KEY: SamplingProvenance("Wan21", torch.zeros(1, 16, 4, 4))}
    with pytest.raises(ValueError, match="does not match"):
        latent_upscale_models.resolve_latent_upscale_model(
            latent, "custom/upscaler_Flux.safetensors"
        )


@pytest.mark.parametrize("latent_format", (None, "SD15", "UnknownLatentFormat"))
@pytest.mark.parametrize("shape", ((2, 16, 4, 6), (1, 16, 2, 4, 6)))
def test_auto_unknown_latents_use_native_bislerp_without_model_io(
    monkeypatch: pytest.MonkeyPatch,
    latent_format: str | None,
    shape: tuple[int, ...],
) -> None:
    """Match native bislerp exactly for unrecognized image and video latents."""
    from simple_syrup.services import latent_upscale_service

    def forbidden(*args: Any) -> Any:
        """Reject checkpoint lookup or downloads on the native fallback path."""
        raise AssertionError("Unknown latents must use native bislerp.")

    monkeypatch.setattr(
        latent_upscale_service, "resolve_latent_upscale_model", forbidden
    )
    samples = torch.randn(shape, generator=torch.Generator().manual_seed(53))
    latent: dict[str, Any] = {
        "samples": samples,
        "noise_mask": torch.ones(1, 4, 6),
        "batch_index": [0],
    }
    if latent_format is not None:
        latent[PROVENANCE_KEY] = SamplingProvenance(latent_format, samples)
    expected = (
        import_module("nodes").LatentUpscaleBy().upscale(latent, "bislerp", 1.5)[0]
    )
    output = LatentUpscaleService().upscale(latent, "Auto", 1.5)
    torch.testing.assert_close(output["samples"], expected["samples"], rtol=0, atol=0)
    assert output["noise_mask"].shape == (1, 6, 9)
    assert output["batch_index"] is latent["batch_index"]
    assert output.get(PROVENANCE_KEY) is latent.get(PROVENANCE_KEY)
    assert latent["samples"] is samples


@pytest.mark.parametrize(
    "latent_format", ("SDXL", "Flux", "Flux2", "Wan21", "Cosmos1CV8x8x8")
)
def test_auto_recognized_latents_keep_learned_model_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, latent_format: str
) -> None:
    """Keep existing learned Auto selection for every supported format."""
    from simple_syrup.services import latent_upscale_service

    samples = torch.ones(1, 4, 4, 6)
    latent = {
        "samples": samples,
        PROVENANCE_KEY: SamplingProvenance(latent_format, samples),
    }
    family = recommended_family(latent_format)
    checkpoint = tmp_path / "upscaler.safetensors"

    def resolve(value: dict[str, Any], method: str) -> tuple[str, Path]:
        """Verify known formats reach the authoritative learned resolver."""
        assert value is latent and method == "Auto"
        return family, checkpoint

    def infer(
        value: torch.Tensor, *, path: Path, family: str, target: tuple[int, int]
    ) -> torch.Tensor:
        """Verify the learned inference boundary receives the chosen checkpoint."""
        assert value is samples and path == checkpoint
        assert family == recommended_family(latent_format) and target == (6, 9)
        return torch.ones(1, 4, *target)

    monkeypatch.setattr(latent_upscale_service, "resolve_latent_upscale_model", resolve)
    monkeypatch.setattr(latent_upscale_service, "upscale_with_sesqui", infer)
    output = LatentUpscaleService().upscale(latent, "Auto", 1.5)
    assert output["samples"].shape == (1, 4, 6, 9)
    assert output[PROVENANCE_KEY] is latent[PROVENANCE_KEY]
