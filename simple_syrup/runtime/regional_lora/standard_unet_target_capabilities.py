# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Classify standard-UNet Linear targets from installed Comfy module ownership."""

from __future__ import annotations

from dataclasses import dataclass

from comfy.ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from torch import nn

from .target_binding import BoundRegionalLoraSpatialCapability


@dataclass(frozen=True, slots=True)
class StandardUnetTargetCapabilities:
    """Retain exact parameter roles and unsupported transformer-block evidence."""

    linear_roles: dict[str, BoundRegionalLoraSpatialCapability]
    unsupported_blocks: tuple[str, ...]


class StandardUnetTargetCapabilityClassifier:
    """Derive regional token roles from installed transformer object identities."""

    def classify(self, graph_root: nn.Module) -> StandardUnetTargetCapabilities:
        """Return exact weight paths without interpreting checkpoint path syntax."""

        if not isinstance(graph_root, nn.Module):
            raise TypeError("Standard UNet capability discovery requires an nn.Module.")
        paths_by_identity = {
            id(module): path for path, module in graph_root.named_modules()
        }
        roles: dict[str, BoundRegionalLoraSpatialCapability] = {}
        unsupported: list[str] = []
        for path, module in graph_root.named_modules():
            if type(module) is not SpatialTransformer:
                continue
            self._classify_transformer(
                module,
                path=path,
                paths_by_identity=paths_by_identity,
                roles=roles,
                unsupported=unsupported,
            )
        return StandardUnetTargetCapabilities(roles, tuple(unsupported))

    def _classify_transformer(
        self,
        transformer: SpatialTransformer,
        *,
        path: str,
        paths_by_identity: dict[int, str],
        roles: dict[str, BoundRegionalLoraSpatialCapability],
        unsupported: list[str],
    ) -> None:
        """Classify one installed spatial transformer and its owned consumers."""

        _register_linear_tree(
            transformer.proj_in,
            BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS,
            paths_by_identity=paths_by_identity,
            roles=roles,
        )
        _register_linear_tree(
            transformer.proj_out,
            BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS,
            paths_by_identity=paths_by_identity,
            roles=roles,
        )
        for index, block in enumerate(transformer.transformer_blocks):
            block_path = f"{path}.transformer_blocks.{index}"
            if type(block) is not BasicTransformerBlock:
                unsupported.append(block_path)
                continue
            self._classify_block(
                block,
                path=block_path,
                paths_by_identity=paths_by_identity,
                roles=roles,
                unsupported=unsupported,
            )

    @staticmethod
    def _classify_block(
        block: BasicTransformerBlock,
        *,
        path: str,
        paths_by_identity: dict[int, str],
        roles: dict[str, BoundRegionalLoraSpatialCapability],
        unsupported: list[str],
    ) -> None:
        """Classify standard image and cross-attention consumers in one block."""

        if block.disable_self_attn or block.switch_temporal_ca_to_sa:
            unsupported.append(path)
            return
        for owner in (block.ff_in, block.ff):
            if isinstance(owner, nn.Module):
                _register_linear_tree(
                    owner,
                    BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS,
                    paths_by_identity=paths_by_identity,
                    roles=roles,
                )
        _register_attention(
            block.attn1,
            query_role=BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS,
            context_role=BoundRegionalLoraSpatialCapability.SPATIAL_TOKENS,
            paths_by_identity=paths_by_identity,
            roles=roles,
        )
        if block.attn2 is None:
            unsupported.append(path)
            return
        _register_attention(
            block.attn2,
            query_role=BoundRegionalLoraSpatialCapability.PACKED_IMAGE_TOKENS,
            context_role=BoundRegionalLoraSpatialCapability.PACKED_CONTEXT_TOKENS,
            paths_by_identity=paths_by_identity,
            roles=roles,
        )


def _register_attention(
    attention: nn.Module,
    *,
    query_role: BoundRegionalLoraSpatialCapability,
    context_role: BoundRegionalLoraSpatialCapability,
    paths_by_identity: dict[int, str],
    roles: dict[str, BoundRegionalLoraSpatialCapability],
) -> None:
    """Register exact Q/output and K/V roles from one attention owner."""

    for owner, role in (
        (attention.to_q, query_role),
        (attention.to_out, query_role),
        (attention.to_k, context_role),
        (attention.to_v, context_role),
    ):
        _register_linear_tree(
            owner,
            role,
            paths_by_identity=paths_by_identity,
            roles=roles,
        )


def _register_linear_tree(
    owner: object,
    role: BoundRegionalLoraSpatialCapability,
    *,
    paths_by_identity: dict[int, str],
    roles: dict[str, BoundRegionalLoraSpatialCapability],
) -> None:
    """Register every exact Linear descendant owned by one known consumer role."""

    if not isinstance(owner, nn.Module):
        return
    for module in owner.modules():
        if not isinstance(module, nn.Linear):
            continue
        path = paths_by_identity.get(id(module))
        if path is None:
            raise ValueError("Transformer Linear owner is absent from the model graph.")
        parameter_path = f"{path}.weight"
        previous = roles.setdefault(parameter_path, role)
        if previous is not role:
            raise ValueError(
                f"Linear target {parameter_path!r} has conflicting consumer roles."
            )


STANDARD_UNET_TARGET_CAPABILITY_CLASSIFIER = StandardUnetTargetCapabilityClassifier()
