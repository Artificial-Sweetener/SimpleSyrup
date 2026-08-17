# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare exact regional variant banks with bounded temporary memory."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Protocol

import torch

from simple_syrup.runtime.regional_lora.standard_unet_variant_materialization import (
    StandardUnetMaterializedVariant,
)

_DEFAULT_CHUNK_ELEMENTS = 1_048_576


class _HashSink(Protocol):
    """Accept ordered bytes for one deterministic digest."""

    def update(self, value: bytes) -> None:
        """Add one byte segment to the digest."""


@dataclass(frozen=True, slots=True)
class MaterializedVariantComparison:
    """Summarize exact structure, bytes, and bounded numerical divergence."""

    region_index: int
    parameter_count: int
    element_count: int
    differing_element_count: int
    max_absolute_error: float
    mean_absolute_error: float
    root_mean_squared_error: float
    max_error_parameter_path: str | None
    reference_sha256: str
    candidate_sha256: str

    @property
    def exact(self) -> bool:
        """Report byte-identical structure and tensor content."""

        return self.reference_sha256 == self.candidate_sha256


def compare_materialized_variants(
    reference: StandardUnetMaterializedVariant,
    candidate: StandardUnetMaterializedVariant,
    *,
    chunk_elements: int = _DEFAULT_CHUNK_ELEMENTS,
) -> MaterializedVariantComparison:
    """Compare aligned parameters incrementally after bounded CPU transfer."""

    _validate_inputs(reference, candidate, chunk_elements=chunk_elements)
    reference_digest = hashlib.sha256()
    candidate_digest = hashlib.sha256()
    _update_bank_metadata(reference_digest, reference)
    _update_bank_metadata(candidate_digest, candidate)
    element_count = 0
    differing_element_count = 0
    maximum_error = 0.0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    maximum_error_path: str | None = None
    with torch.no_grad():
        for reference_parameter, candidate_parameter in zip(
            reference.parameters,
            candidate.parameters,
            strict=True,
        ):
            _validate_parameter_pair(reference_parameter, candidate_parameter)
            reference_cpu = reference_parameter.tensor.detach().to("cpu")
            candidate_cpu = candidate_parameter.tensor.detach().to("cpu")
            _update_parameter_metadata(
                reference_digest,
                reference_parameter.path,
                reference_cpu,
            )
            _update_parameter_metadata(
                candidate_digest,
                candidate_parameter.path,
                candidate_cpu,
            )
            reference_flat = reference_cpu.contiguous().reshape(-1)
            candidate_flat = candidate_cpu.contiguous().reshape(-1)
            for start in range(0, reference_flat.numel(), chunk_elements):
                stop = min(start + chunk_elements, reference_flat.numel())
                reference_chunk = reference_flat[start:stop]
                candidate_chunk = candidate_flat[start:stop]
                _require_finite(reference_chunk, reference_parameter.path)
                _require_finite(candidate_chunk, candidate_parameter.path)
                reference_digest.update(_raw_tensor_bytes(reference_chunk))
                candidate_digest.update(_raw_tensor_bytes(candidate_chunk))
                differing_element_count += int(
                    torch.count_nonzero(reference_chunk != candidate_chunk).item()
                )
                delta = candidate_chunk.to(torch.float64) - reference_chunk.to(
                    torch.float64
                )
                absolute_delta = delta.abs()
                local_maximum = float(absolute_delta.max().item())
                if local_maximum > maximum_error:
                    maximum_error = local_maximum
                    maximum_error_path = reference_parameter.path
                absolute_error_sum += float(absolute_delta.sum().item())
                squared_error_sum += float(torch.square(delta).sum().item())
                element_count += reference_chunk.numel()
    if element_count < 1:
        raise ValueError("Materialized variant comparison requires tensor elements.")
    return MaterializedVariantComparison(
        region_index=reference.region_index,
        parameter_count=len(reference.parameters),
        element_count=element_count,
        differing_element_count=differing_element_count,
        max_absolute_error=maximum_error,
        mean_absolute_error=absolute_error_sum / element_count,
        root_mean_squared_error=math.sqrt(squared_error_sum / element_count),
        max_error_parameter_path=maximum_error_path,
        reference_sha256=reference_digest.hexdigest(),
        candidate_sha256=candidate_digest.hexdigest(),
    )


def _validate_inputs(
    reference: object,
    candidate: object,
    *,
    chunk_elements: int,
) -> None:
    """Require comparable bank values and a bounded positive chunk size."""

    if not isinstance(reference, StandardUnetMaterializedVariant) or not isinstance(
        candidate, StandardUnetMaterializedVariant
    ):
        raise TypeError("Materialized comparison requires two variant banks.")
    if reference.region_index != candidate.region_index:
        raise ValueError("Materialized variant region indices do not match.")
    if isinstance(chunk_elements, bool) or not isinstance(chunk_elements, int):
        raise TypeError("Materialized comparison chunk size must be an integer.")
    if chunk_elements < 1:
        raise ValueError("Materialized comparison chunk size must be positive.")
    reference_paths = tuple(parameter.path for parameter in reference.parameters)
    candidate_paths = tuple(parameter.path for parameter in candidate.parameters)
    if reference_paths != candidate_paths:
        raise ValueError("Materialized variant parameter paths do not match.")


def _validate_parameter_pair(reference: object, candidate: object) -> None:
    """Require one aligned shape and dtype without device restrictions."""

    reference_tensor = getattr(reference, "tensor", None)
    candidate_tensor = getattr(candidate, "tensor", None)
    if not isinstance(reference_tensor, torch.Tensor) or not isinstance(
        candidate_tensor, torch.Tensor
    ):
        raise TypeError("Materialized comparison parameters must contain tensors.")
    if reference_tensor.shape != candidate_tensor.shape:
        raise ValueError("Materialized variant parameter shapes do not match.")
    if reference_tensor.dtype != candidate_tensor.dtype:
        raise ValueError("Materialized variant parameter dtypes do not match.")


def _update_bank_metadata(
    digest: _HashSink,
    variant: StandardUnetMaterializedVariant,
) -> None:
    """Frame one bank identity independently of object or device identity."""

    digest.update(struct.pack("<qQ", variant.region_index, len(variant.parameters)))


def _update_parameter_metadata(
    digest: _HashSink,
    path: str,
    tensor: torch.Tensor,
) -> None:
    """Frame path, dtype, shape, and element count before raw tensor bytes."""

    _update_string(digest, path)
    _update_string(digest, str(tensor.dtype))
    digest.update(struct.pack("<Q", tensor.ndim))
    for dimension in tensor.shape:
        digest.update(struct.pack("<q", dimension))
    digest.update(struct.pack("<Q", tensor.numel()))


def _update_string(digest: _HashSink, value: str) -> None:
    """Add one length-prefixed UTF-8 string to a bank digest."""

    encoded = value.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)


def _raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return device-neutral contiguous storage bytes for one CPU chunk."""

    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _require_finite(tensor: torch.Tensor, path: str) -> None:
    """Reject nonfinite weights whose error metrics would be undefined."""

    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(
            f"Materialized variant parameter {path!r} must contain finite values."
        )
