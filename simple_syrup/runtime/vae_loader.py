# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load ComfyUI VAE selections using ComfyUI's VAE loader policy."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

VIDEO_TAES = ("taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2")
IMAGE_TAES = ("taesd", "taesdxl", "taesd3", "taef1")


class VaeLoaderService:
    """Load VAE choices through ComfyUI-compatible selection rules."""

    def __init__(self, folder_paths_module: ModuleType | None = None) -> None:
        """Create a VAE loader with injectable ComfyUI folder paths."""

        self._folder_paths_module = folder_paths_module

    def load_vae(self, vae_name: str) -> object:
        """Load and validate the selected VAE."""

        if vae_name == "pixel_space":
            sd: dict[str, object] = {"pixel_space_vae": torch.tensor(1.0)}
            metadata = None
        elif vae_name in IMAGE_TAES:
            sd = _load_taesd(vae_name, self._folder_paths())
            metadata = None
        else:
            folder_name = (
                "vae_approx" if os.path.splitext(vae_name)[0] in VIDEO_TAES else "vae"
            )
            vae_path = self._folder_paths().get_full_path_or_raise(
                folder_name,
                vae_name,
            )
            return load_vae_path(Path(str(vae_path)))

        return _build_vae(sd, metadata)

    def _folder_paths(self) -> ModuleType:
        """Return the ComfyUI folder_paths module."""

        if self._folder_paths_module is not None:
            return self._folder_paths_module
        module = _folder_paths()
        self._folder_paths_module = module
        return module


def vae_choices(folder_paths_module: ModuleType | None = None) -> list[str]:
    """Return VAE choices matching ComfyUI's `VAELoader` list policy."""

    folder_paths = folder_paths_module or _folder_paths()
    vaes: list[str] = list(folder_paths.get_filename_list("vae"))
    approx_vaes: list[str] = list(folder_paths.get_filename_list("vae_approx"))
    sdxl_taesd_enc = False
    sdxl_taesd_dec = False
    sd1_taesd_enc = False
    sd1_taesd_dec = False
    sd3_taesd_enc = False
    sd3_taesd_dec = False
    f1_taesd_enc = False
    f1_taesd_dec = False

    for vae in approx_vaes:
        if vae.startswith("taesd_decoder."):
            sd1_taesd_dec = True
        elif vae.startswith("taesd_encoder."):
            sd1_taesd_enc = True
        elif vae.startswith("taesdxl_decoder."):
            sdxl_taesd_dec = True
        elif vae.startswith("taesdxl_encoder."):
            sdxl_taesd_enc = True
        elif vae.startswith("taesd3_decoder."):
            sd3_taesd_dec = True
        elif vae.startswith("taesd3_encoder."):
            sd3_taesd_enc = True
        elif vae.startswith("taef1_encoder."):
            f1_taesd_dec = True
        elif vae.startswith("taef1_decoder."):
            f1_taesd_enc = True
        else:
            for tae in VIDEO_TAES:
                if vae.startswith(tae):
                    vaes.append(vae)

    if sd1_taesd_dec and sd1_taesd_enc:
        vaes.append("taesd")
    if sdxl_taesd_dec and sdxl_taesd_enc:
        vaes.append("taesdxl")
    if sd3_taesd_dec and sd3_taesd_enc:
        vaes.append("taesd3")
    if f1_taesd_dec and f1_taesd_enc:
        vaes.append("taef1")
    vaes.append("pixel_space")
    return vaes


def load_vae_path(path: Path) -> object:
    """Load a conventional VAE file from an absolute path."""

    comfy_utils = importlib.import_module("comfy.utils")
    sd, metadata = comfy_utils.load_torch_file(str(path), return_metadata=True)
    return _build_vae(sd, metadata)


def _load_taesd(name: str, folder_paths: ModuleType) -> dict[str, object]:
    """Load a TAESD encoder and decoder pair using ComfyUI's naming policy."""

    sd: dict[str, object] = {}
    approx_vaes: list[str] = list(folder_paths.get_filename_list("vae_approx"))
    encoder = next(vae for vae in approx_vaes if vae.startswith(f"{name}_encoder."))
    decoder = next(vae for vae in approx_vaes if vae.startswith(f"{name}_decoder."))
    comfy_utils = importlib.import_module("comfy.utils")

    enc = comfy_utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae_approx", encoder)
    )
    for key, value in enc.items():
        sd[f"taesd_encoder.{key}"] = value

    dec = comfy_utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae_approx", decoder)
    )
    for key, value in dec.items():
        sd[f"taesd_decoder.{key}"] = value

    if name == "taesd":
        sd["vae_scale"] = torch.tensor(0.18215)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesdxl":
        sd["vae_scale"] = torch.tensor(0.13025)
        sd["vae_shift"] = torch.tensor(0.0)
    elif name == "taesd3":
        sd["vae_scale"] = torch.tensor(1.5305)
        sd["vae_shift"] = torch.tensor(0.0609)
    elif name == "taef1":
        sd["vae_scale"] = torch.tensor(0.3611)
        sd["vae_shift"] = torch.tensor(0.1159)
    return sd


def _build_vae(sd: dict[str, object], metadata: object | None) -> object:
    """Instantiate and validate a ComfyUI VAE object."""

    comfy_sd = _comfy_sd()
    vae = comfy_sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def _folder_paths() -> ModuleType:
    """Import ComfyUI folder paths lazily."""

    module: Any = importlib.import_module("folder_paths")
    if not isinstance(module, ModuleType):
        raise TypeError("folder_paths import did not return a module.")
    return module


def _comfy_sd() -> Any:
    """Import ComfyUI's stable diffusion loading module lazily."""

    return importlib.import_module("comfy.sd")
