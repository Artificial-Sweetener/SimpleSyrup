# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for latent diagnostics report generation."""

from __future__ import annotations

import torch

from simple_syrup.services.latent_diagnostics_service import LatentDiagnosticsService


def test_describe_reports_bchw_latent_as_compatible() -> None:
    """A standard ComfyUI BCHW latent is reported as MoD-compatible."""

    latent = {
        "samples": torch.zeros((2, 4, 8, 16), dtype=torch.float16),
        "batch_index": [0, 1],
    }

    report = LatentDiagnosticsService().describe(latent)

    assert "latent_keys: [batch_index, samples]" in report
    assert "shape: [2, 4, 8, 16]" in report
    assert "ndim: 4" in report
    assert "dtype: torch.float16" in report
    assert "height: 8" in report
    assert "width: 16" in report
    assert "mixture_of_diffusers_current_compatible: yes" in report
    assert "batch_index: builtins.list len=2" in report


def test_describe_reports_non_4d_tensor_as_incompatible() -> None:
    """A tensor with spatial dimensions but no BCHW layout is called out clearly."""

    latent = {"samples": torch.zeros((4, 8, 16), dtype=torch.float32)}

    report = LatentDiagnosticsService().describe(latent)

    assert "shape: [4, 8, 16]" in report
    assert "ndim: 3" in report
    assert "spatial_last_dims:" in report
    assert "bchw_interpretation: unavailable" in report
    assert "mixture_of_diffusers_current_compatible: no" in report
    assert "expects non-nested 4D BCHW or singleton-depth 5D BCDHW samples" in report


def test_describe_reports_singleton_depth_5d_latent_as_compatible() -> None:
    """An Anima-style singleton-depth BCDHW latent is reported as compatible."""

    latent = {"samples": torch.zeros((1, 16, 1, 252, 180), dtype=torch.float32)}

    report = LatentDiagnosticsService().describe(latent)

    assert "shape: [1, 16, 1, 252, 180]" in report
    assert "ndim: 5" in report
    assert "bcdhw_interpretation:" in report
    assert "depth: 1" in report
    assert "height: 252" in report
    assert "width: 180" in report
    assert "mixture_of_diffusers_current_compatible: yes" in report


def test_describe_reports_non_singleton_depth_5d_latent_as_incompatible() -> None:
    """A non-singleton BCDHW latent remains unsupported until validated."""

    latent = {"samples": torch.zeros((1, 16, 2, 8, 8), dtype=torch.float32)}

    report = LatentDiagnosticsService().describe(latent)

    assert "bcdhw_interpretation:" in report
    assert "depth: 2" in report
    assert "mixture_of_diffusers_current_compatible: no" in report
    assert "expects 5D samples to use a singleton depth axis" in report


def test_describe_reports_missing_samples_without_crashing() -> None:
    """A malformed latent dictionary receives an actionable report."""

    report = LatentDiagnosticsService().describe({"noise_mask": torch.ones((1, 8, 8))})

    assert "samples: missing or not a torch.Tensor" in report
    assert "samples_type: builtins.NoneType" in report
    assert "compatibility_reason: samples must be a torch.Tensor." in report
    assert "noise_mask: torch.Tensor shape=[1, 8, 8]" in report
