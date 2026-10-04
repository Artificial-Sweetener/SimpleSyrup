"""List checksum-pinned SesquiLSR artifacts in Comfy's standard latent model folder."""

from __future__ import annotations

from importlib import import_module

from .auto_model_artifact import AutoModelArtifact

SESQUI_REVISION = "befae004248c403f38b76b9f65fd43b901ea3eaa"
SESQUI_REPOSITORY = "https://github.com/LoganBooker/SesquiLSR"
_CHECKPOINTS = (
    (
        "wan",
        "Wan21",
        12373048,
        "d1b4c2c1d80b2af81e6bbfd38501ffb5adffb9f7c0ed2599555b837d0d2d015a",
    ),
    (
        "sdxl",
        "SDXL",
        12293696,
        "392c0e36fc0389a170cf6c1682556972d91d750c31aa392cdad1b13cad4e18e4",
    ),
    (
        "flux",
        "Flux",
        12374768,
        "dbc6362c910a179a22eca26f656fdacde17eded876ee8f05b8f5b73630e4ddbe",
    ),
    (
        "flux2",
        "Flux2",
        12525400,
        "f385ff2250b3f2c72ad422768558414d617b9a006cc0b1ce37a296cc99d17177",
    ),
)
LATENT_UPSCALE_ARTIFACTS = {
    family: AutoModelArtifact(
        cache_id=f"sesqui_lsr_{family}_{SESQUI_REVISION}",
        filename=f"upscaler_{name}.safetensors",
        folder_name="latent_upscale_models",
        canonical_subfolder=family,
        source_url=(
            f"https://raw.githubusercontent.com/LoganBooker/SesquiLSR/"
            f"{SESQUI_REVISION}/models/upscaler_{name}.safetensors"
        ),
        source_repo=SESQUI_REPOSITORY,
        description=f"SesquiLSR {name} latent upscaler",
        sha256=checksum,
        file_size_bytes=size,
    )
    for family, name, size, checksum in _CHECKPOINTS
}


def latent_model_choices() -> list[str]:
    """Advertise pinned downloads and installed checkpoints without hashing or
    IO loading.
    """
    from ..domain.latent_upscale import NATIVE_LATENT_METHODS

    known = [
        f"{artifact.canonical_subfolder}/{artifact.filename}"
        for artifact in LATENT_UPSCALE_ARTIFACTS.values()
    ]
    installed = [
        name.replace("\\", "/")
        for name in import_module("folder_paths").get_filename_list(
            "latent_upscale_models"
        )
    ]
    return list(dict.fromkeys(["Auto", *known, *installed, *NATIVE_LATENT_METHODS]))
