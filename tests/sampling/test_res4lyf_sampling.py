# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove pinned RES4LYF method registration and solver binding parity."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from simple_syrup.runtime.res4lyf_sampler_names import RES4LYF_SAMPLER_NAMES
from simple_syrup.runtime.sampling_noise import prepare_sampling_noise
from simple_syrup.runtime.sampling_samplers import available_samplers, resolve_sampler
from simple_syrup.runtime.sampling_schedulers import (
    available_schedulers,
    calculate_sigmas,
)
from simple_syrup.third_party.res4lyf_runtime import sigmas as upstream_sigmas
from simple_syrup.third_party.res4lyf_runtime.beta.rk_coefficients_beta import (
    RK_SAMPLER_NAMES_BETA_FOLDERS,
    process_sampler_name,
)

SOURCE_REVISION = "3d1d69da69ee47f7647d59e1bd0967e472fccc41"
SOLVER_SOURCE_SHA256 = (
    "7e1cd8c1be32b4ec0f8efe827954ab96b529143fef00915f1faa7034908f8f88"
)
SOLVER_SOURCE_FILES = (
    "helper.py",
    "latents.py",
    "style_transfer.py",
    "sigmas.py",
    "beta/constants.py",
    "beta/deis_coefficients.py",
    "beta/phi_functions.py",
    "beta/rk_coefficients_beta.py",
    "beta/rk_method_beta.py",
    "beta/noise_classes.py",
    "beta/rk_noise_sampler_beta.py",
    "beta/rk_guide_func_beta.py",
    "beta/rk_sampler_beta.py",
)


def test_pinned_solver_source_matches_upstream_with_one_recorded_fix() -> None:
    """Guard the solver copy and its single upstream error correction."""

    root = (
        Path(__file__).resolve().parents[2]
        / "simple_syrup"
        / "third_party"
        / "res4lyf_runtime"
    )
    digest = hashlib.sha256()
    for relative_path in SOLVER_SOURCE_FILES:
        digest.update(relative_path.encode())
        source = (root / relative_path).read_bytes().replace(b"\r\n", b"\n")
        if relative_path == "beta/rk_coefficients_beta.py":
            branch = source.index(b'case "res_8s_alt"')
            corrected = b"            ci = [c1, c2, c3, c4, c5, c6, c7, c8]"
            original = b"            #ci = [c1, c2, c3, c4, c5, c6, c7, c8]"
            assert source[branch:].count(corrected) >= 1
            source = source[:branch] + source[branch:].replace(corrected, original, 1)
        digest.update(source)
    assert digest.hexdigest() == SOLVER_SOURCE_SHA256


def test_res4lyf_menu_covers_every_upstream_solver_method() -> None:
    """Expose each pinned upstream menu method exactly once."""

    assert RES4LYF_SAMPLER_NAMES == tuple(RK_SAMPLER_NAMES_BETA_FOLDERS[1:])
    assert len(RES4LYF_SAMPLER_NAMES) == 118
    assert len(set(RES4LYF_SAMPLER_NAMES)) == 118
    assert Counter(name.split("/", 1)[0] for name in RES4LYF_SAMPLER_NAMES) == {
        "exponential": 41,
        "fully_implicit": 30,
        "linear": 20,
        "multistep": 10,
        "hybrid": 9,
        "diag_implicit": 8,
    }
    assert set(RES4LYF_SAMPLER_NAMES).issubset(available_samplers())


def test_each_res4lyf_menu_name_binds_the_original_solver_and_method() -> None:
    """Keep the method and implicit selection identical to RES4LYF."""

    for name in RES4LYF_SAMPLER_NAMES:
        sampler = resolve_sampler(name)
        expected_method, expected_implicit = process_sampler_name(name)
        assert vars(sampler)["sampler_function"].__name__ == "_sample_res4lyf"
        assert vars(sampler)["extra_options"] == {
            "rk_type": expected_method,
            "implicit_sampler_name": expected_implicit,
            "implicit_type": "bongmath",
            "implicit_type_substeps": "bongmath",
            "bongmath": name != "linear/rk5_7s",
        }


def test_bong_tangent_uses_original_res4lyf_sigma_function() -> None:
    """Preserve RES4LYF's schedule values and local dropdown entry."""

    class Model:
        """Provide the scheduler's model-sampling reference."""

        def get_model_object(self, name: str) -> object:
            """Return the model-sampling object for sigma generation."""

            assert name == "model_sampling"
            return SimpleNamespace()

    model = Model()
    expected = upstream_sigmas.bong_tangent_scheduler(
        model.get_model_object("model_sampling"), 8
    )
    actual = calculate_sigmas(model, "bong_tangent", "euler", 8, 1.0)
    assert "bong_tangent" in available_schedulers()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_res4lyf_default_noise_matches_upstream_reference() -> None:
    """Match upstream Gaussian generation and channel normalization exactly."""

    class Model:
        """Provide the same model sigma bounds as the upstream reference."""

        def get_model_object(self, name: str) -> object:
            """Return the model-sampling bounds."""

            assert name == "model_sampling"
            return SimpleNamespace(sigma_max=1.0, sigma_min=0.0)

    noise = prepare_sampling_noise(
        comfy_sample=SimpleNamespace(),
        sampler_name="exponential/ddim",
        samples=torch.zeros((1, 4, 8, 8)),
        seed=42,
        batch_indices=None,
        model=Model(),
    )
    assert hashlib.sha256(noise.numpy().tobytes()).hexdigest() == (
        "c4e80cfdfbbd88ad1dacfeacf19698188a4870ee2b17122d3e461001f6dce8d5"
    )


def test_core_sampler_keeps_comfy_noise_path() -> None:
    """Leave ComfyUI's seed and batch-index behavior intact for core methods."""

    samples = torch.zeros((1, 4, 8, 8))
    expected = torch.ones_like(samples)
    calls: list[tuple[torch.Tensor, int, list[int]]] = []

    class Model:
        """Provide the sampling protocol without using it for core noise."""

        def get_model_object(self, name: str) -> object:
            """Reject model lookups on the core ComfyUI noise path."""

            raise AssertionError(name)

    def prepare_noise(
        input_samples: torch.Tensor, seed: int, batch_indices: list[int]
    ) -> torch.Tensor:
        """Record the arguments delegated to ComfyUI."""

        calls.append((input_samples, seed, batch_indices))
        return expected

    actual = prepare_sampling_noise(
        comfy_sample=SimpleNamespace(prepare_noise=prepare_noise),
        sampler_name="euler",
        samples=samples,
        seed=42,
        batch_indices=[3],
        model=Model(),
    )
    assert actual is expected
    assert calls == [(samples, 42, [3])]


def test_every_res4lyf_method_executes_with_finite_latents() -> None:
    """Exercise every advertised solver on a controlled two-step flow model."""

    class ModelSampling:
        """Provide the sigma bounds and denoising rule used by the solver."""

        sigma_max = torch.tensor(1.0)
        sigma_min = torch.tensor(0.01)

        def calculate_denoised(
            self, sigma: torch.Tensor, eps: torch.Tensor, x: torch.Tensor
        ) -> torch.Tensor:
            """Return a linear test denoising prediction."""

            return x - sigma * eps

    class Model:
        """Expose the ComfyUI model shape expected by the pinned solver."""

        def __init__(self) -> None:
            """Create a small model surface without external weights."""

            inner = SimpleNamespace(
                device=torch.device("cpu"),
                model_sampling=ModelSampling(),
                diffusion_model=SimpleNamespace(),
            )
            self.inner_model = SimpleNamespace(inner_model=inner)

        def __call__(
            self, x: torch.Tensor, sigma: torch.Tensor, **kwargs: object
        ) -> torch.Tensor:
            """Return a deterministic prediction for every solver stage."""

            return x * 0.5

    model = Model()
    input_latent = torch.ones((1, 4, 8, 8))
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    for name in RES4LYF_SAMPLER_NAMES:
        sampler = resolve_sampler(name)
        sample = vars(sampler)["sampler_function"]
        output = sample(
            model,
            input_latent.clone(),
            sigmas,
            extra_args={
                "seed": 42,
                "model_options": {"transformer_options": {}},
            },
            callback=None,
            disable=True,
            **vars(sampler)["extra_options"],
        )
        assert output.shape == input_latent.shape, name
        assert torch.isfinite(output).all(), name
