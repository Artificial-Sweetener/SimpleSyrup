# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for vendored third-party provenance."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_third_party_manifest_references_existing_licenses_and_runtime_files() -> None:
    """Vendored components should record license text and runtime file paths."""

    manifest = tomllib.loads(
        (REPO_ROOT / "third_party" / "manifest.toml").read_text(encoding="utf-8")
    )

    for component in manifest["component"]:
        license_path = REPO_ROOT / component["license_file"]
        assert license_path.is_file()

        for vendored_file in component["vendored_files"]:
            assert (REPO_ROOT / vendored_file).is_file()


def test_third_party_notice_records_every_manifest_component() -> None:
    """The notice file should identify each vendored component by name."""

    manifest = tomllib.loads(
        (REPO_ROOT / "third_party" / "manifest.toml").read_text(encoding="utf-8")
    )
    notice = (REPO_ROOT / "third_party" / "NOTICE.md").read_text(encoding="utf-8")

    for component in manifest["component"]:
        assert component["name"] in notice


def test_adapted_owned_files_point_to_third_party_provenance() -> None:
    """Mixed-provenance owned files should carry a local third-party notice."""

    manifest = tomllib.loads(
        (REPO_ROOT / "third_party" / "manifest.toml").read_text(encoding="utf-8")
    )

    for component in manifest["component"]:
        for vendored_file in component["vendored_files"]:
            path = Path(vendored_file)
            if path.parts[:2] == ("simple_syrup", "third_party"):
                continue

            content = (REPO_ROOT / path).read_text(encoding="utf-8")

            assert "third_party/manifest.toml" in content, path
            assert "third_party/NOTICE.md" in content, path


def test_res4lyf_beta57_provenance_is_recorded() -> None:
    """The beta57 scheduler preset should trace to the inspected RES4LYF source."""

    manifest = tomllib.loads(
        (REPO_ROOT / "third_party" / "manifest.toml").read_text(encoding="utf-8")
    )
    components = {component["name"]: component for component in manifest["component"]}

    res4lyf_beta57 = components["RES4LYF beta57 scheduler preset"]
    license_path = REPO_ROOT / res4lyf_beta57["license_file"]

    assert res4lyf_beta57["license"] == "AGPL-3.0"
    assert "GNU AFFERO GENERAL PUBLIC" in license_path.read_text(encoding="utf-8")
    assert res4lyf_beta57["source"] == "https://github.com/ClownsharkBatwing/RES4LYF"
    assert res4lyf_beta57["revision"] == "1c9bf61"
    assert res4lyf_beta57["source_paths"] == [
        "sigmas.py",
        "res4lyf.py",
        "README.md",
    ]
    assert res4lyf_beta57["vendored_files"] == [
        "simple_syrup/runtime/sampling_schedulers.py",
    ]


def test_automatic1111_sampler_integration_provenance_is_recorded() -> None:
    """The A1111 sampler integration should trace to the inspected WebUI source."""

    manifest = tomllib.loads(
        (REPO_ROOT / "third_party" / "manifest.toml").read_text(encoding="utf-8")
    )
    components = {component["name"]: component for component in manifest["component"]}

    automatic1111 = components["AUTOMATIC1111 Euler a sampler integration"]
    license_path = REPO_ROOT / automatic1111["license_file"]

    assert automatic1111["license"] == "AGPL-3.0"
    assert "GNU AFFERO GENERAL PUBLIC" in license_path.read_text(encoding="utf-8")
    assert (
        automatic1111["source"]
        == "https://github.com/AUTOMATIC1111/stable-diffusion-webui"
    )
    assert automatic1111["revision"] == "0120768f"
    assert automatic1111["source_paths"] == [
        "modules/sd_samplers_kdiffusion.py",
        "modules/sd_samplers_common.py",
        "modules/sd_schedulers.py",
        "modules/rng.py",
    ]
    assert automatic1111["vendored_files"] == [
        "simple_syrup/runtime/a1111_sampling.py",
        "simple_syrup/runtime/sampling_samplers.py",
        "simple_syrup/runtime/sampling_schedulers.py",
    ]


def test_k_diffusion_euler_ancestral_provenance_is_recorded() -> None:
    """The A1111 sampler loop should trace to the inspected k-diffusion source."""

    manifest = tomllib.loads(
        (REPO_ROOT / "third_party" / "manifest.toml").read_text(encoding="utf-8")
    )
    components = {component["name"]: component for component in manifest["component"]}

    k_diffusion = components["k-diffusion Euler ancestral sampler"]
    license_path = REPO_ROOT / k_diffusion["license_file"]

    assert k_diffusion["license"] == "MIT"
    assert "Copyright (c) 2022 Katherine Crowson" in license_path.read_text(
        encoding="utf-8"
    )
    assert k_diffusion["source"] == "https://github.com/crowsonkb/k-diffusion"
    assert k_diffusion["revision"] == "ab527a9"
    assert k_diffusion["source_paths"] == [
        "k_diffusion/sampling.py",
    ]
    assert k_diffusion["vendored_files"] == [
        "simple_syrup/runtime/a1111_sampling.py",
    ]


def test_tiled_diffusion_provenance_is_recorded() -> None:
    """The tiled denoising behavior should trace to the inspected extension."""

    manifest = tomllib.loads(
        (REPO_ROOT / "third_party" / "manifest.toml").read_text(encoding="utf-8")
    )
    components = {component["name"]: component for component in manifest["component"]}

    tiled_diffusion = components[
        "Mixture of Diffusers and MultiDiffusion tiled diffusion behavior"
    ]
    license_path = REPO_ROOT / tiled_diffusion["license_file"]

    assert tiled_diffusion["license"] == "CC-BY-NC-SA-4.0"
    assert "Attribution-NonCommercial-ShareAlike 4.0" in license_path.read_text(
        encoding="utf-8"
    )
    assert (
        tiled_diffusion["source"]
        == "https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111"
    )
    assert tiled_diffusion["source_paths"] == [
        "tile_methods/abstractdiffusion.py",
        "tile_methods/mixtureofdiffusers.py",
        "tile_methods/multidiffusion.py",
        "tile_utils/utils.py",
        "scripts/tilediffusion.py",
    ]
    assert tiled_diffusion["vendored_files"] == [
        "simple_syrup/domain/regional_detailing.py",
        "simple_syrup/domain/tiled_diffusion.py",
        "simple_syrup/masking/regional_detailing_masks.py",
        "simple_syrup/runtime/mixture_of_diffusers_sampling.py",
        "simple_syrup/runtime/multidiffusion_sampling.py",
        "simple_syrup/runtime/regional_multidiffusion_sampling.py",
        "simple_syrup/runtime/tiled_sampling.py",
        "simple_syrup/services/detail_segs_as_regions_service.py",
    ]


def test_notice_records_sampler_and_tiled_diffusion_provenance() -> None:
    """The notice file should include sampler and tiled diffusion provenance."""

    notice = (REPO_ROOT / "third_party" / "NOTICE.md").read_text(encoding="utf-8")

    assert "AUTOMATIC1111 Euler a sampler integration" in notice
    assert "k-diffusion Euler ancestral sampler" in notice
    assert "Mixture of Diffusers and MultiDiffusion tiled diffusion behavior" in notice
    assert "regional prompt mask blending" in notice
