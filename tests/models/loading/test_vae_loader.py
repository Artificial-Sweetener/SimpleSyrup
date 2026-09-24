# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for shared ComfyUI-compatible VAE loading."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from simple_syrup.runtime.vae_loader import VaeLoaderService, vae_choices


@dataclass
class FakeComfyState:
    """Recorded calls into fake ComfyUI VAE APIs."""

    loaded_paths: list[str] = field(default_factory=list)
    vae_states: list[dict[str, object]] = field(default_factory=list)
    vae_metadata: list[object | None] = field(default_factory=list)
    validations: int = 0


class FakeFolderPaths(ModuleType):
    """Folder paths fake with VAE model methods."""

    def __init__(self, models_dir: Path) -> None:
        """Create fake VAE folder state."""

        super().__init__("folder_paths")
        self.models_dir = str(models_dir)
        self.files: dict[str, list[str]] = {
            "vae": ["manual_vae.safetensors"],
            "vae_approx": [],
        }

    def get_filename_list(self, folder_name: str) -> list[str]:
        """Return configured relative filenames."""

        return self.files.get(folder_name, [])

    def get_full_path_or_raise(self, folder_name: str, filename: str) -> str:
        """Return a deterministic absolute path for a relative filename."""

        return str(Path(self.models_dir) / folder_name / filename)


def test_vae_choices_match_comfy_special_choices(tmp_path: Path) -> None:
    """VAE choices include ComfyUI TAESD, video TAE, and pixel-space entries."""

    folder_paths = FakeFolderPaths(tmp_path / "models")
    folder_paths.files["vae"] = ["manual_vae.safetensors"]
    folder_paths.files["vae_approx"] = [
        "taesd_encoder.pth",
        "taesd_decoder.pth",
        "taehv.pth",
    ]

    assert vae_choices(folder_paths) == [
        "manual_vae.safetensors",
        "taehv.pth",
        "taesd",
        "pixel_space",
    ]


def test_vae_loader_loads_pixel_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pixel-space VAE selection builds and validates a ComfyUI VAE."""

    state = _install_fake_comfy(monkeypatch)
    service = VaeLoaderService(FakeFolderPaths(tmp_path / "models"))

    result = service.load_vae("pixel_space")

    assert result is not None
    assert "pixel_space_vae" in state.vae_states[0]
    assert state.vae_metadata == [None]
    assert state.validations == 1


def test_vae_loader_loads_manual_vae_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manual VAE files load from the ComfyUI `vae` folder."""

    state = _install_fake_comfy(monkeypatch)
    folder_paths = FakeFolderPaths(tmp_path / "models")
    service = VaeLoaderService(folder_paths)

    service.load_vae("manual_vae.safetensors")

    assert state.loaded_paths == [
        str(tmp_path / "models" / "vae" / "manual_vae.safetensors")
    ]
    assert state.vae_states[0]["loaded_from"] == state.loaded_paths[0]
    assert state.vae_metadata[0] == {"metadata_from": state.loaded_paths[0]}


def test_vae_loader_loads_video_tae_from_approx_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video TAE files load from ComfyUI's `vae_approx` folder."""

    state = _install_fake_comfy(monkeypatch)
    folder_paths = FakeFolderPaths(tmp_path / "models")
    folder_paths.files["vae_approx"] = ["taehv.pth"]
    service = VaeLoaderService(folder_paths)

    service.load_vae("taehv.pth")

    assert state.loaded_paths == [str(tmp_path / "models" / "vae_approx" / "taehv.pth")]


def test_vae_loader_loads_taesd_encoder_decoder_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TAESD selections load matching encoder and decoder files."""

    state = _install_fake_comfy(monkeypatch)
    folder_paths = FakeFolderPaths(tmp_path / "models")
    folder_paths.files["vae_approx"] = [
        "taesd_encoder.pth",
        "taesd_decoder.pth",
    ]
    service = VaeLoaderService(folder_paths)

    service.load_vae("taesd")

    assert state.loaded_paths == [
        str(tmp_path / "models" / "vae_approx" / "taesd_encoder.pth"),
        str(tmp_path / "models" / "vae_approx" / "taesd_decoder.pth"),
    ]
    assert state.vae_states[0]["taesd_encoder.loaded_from"] == state.loaded_paths[0]
    assert state.vae_states[0]["taesd_decoder.loaded_from"] == state.loaded_paths[1]
    assert "vae_scale" in state.vae_states[0]
    assert "vae_shift" in state.vae_states[0]


def _install_fake_comfy(monkeypatch: pytest.MonkeyPatch) -> FakeComfyState:
    """Install fake ComfyUI modules and return recorded VAE state."""

    class FakeVAE:
        """Small VAE fake that records state dicts and validation."""

        def __init__(
            self,
            sd: dict[str, object],
            metadata: object | None = None,
        ) -> None:
            """Record constructor inputs."""

            state.vae_states.append(sd)
            state.vae_metadata.append(metadata)

        def throw_exception_if_invalid(self) -> None:
            """Accept validation and record that it happened."""

            state.validations += 1

    comfy_module = ModuleType("comfy")
    comfy_sd = ModuleType("comfy.sd")
    comfy_utils = ModuleType("comfy.utils")
    state = FakeComfyState()

    def load_torch_file(
        path: str,
        return_metadata: bool = False,
    ) -> dict[str, object] | tuple[dict[str, object], dict[str, object]]:
        """Record VAE file loads and return fake torch state."""

        state.loaded_paths.append(path)
        sd: dict[str, object] = {"loaded_from": path}
        if return_metadata:
            return sd, {"metadata_from": path}
        return sd

    comfy_sd.VAE = FakeVAE  # type: ignore[attr-defined]
    comfy_utils.load_torch_file = load_torch_file  # type: ignore[attr-defined]
    comfy_module.sd = comfy_sd  # type: ignore[attr-defined]
    comfy_module.utils = comfy_utils  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", comfy_sd)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils)
    return state
