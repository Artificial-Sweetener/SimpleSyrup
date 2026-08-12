# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discover and validate the installed Anima diffusion module surface."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

from comfy.ldm.anima.model import Anima
from comfy.ldm.cosmos.predict2 import Attention, Block, GPT2FeedForward
from torch import nn

from .anima_targets import (
    ANIMA_BLOCK_COUNT,
    AnimaLoraTargetFamily,
    anima_lora_target_name,
    expected_anima_lora_features,
)

_REQUIRED = object()
_OwnerType = TypeVar("_OwnerType", bound=nn.Module)


@dataclass(frozen=True)
class _ParameterContract:
    """Describe one required bound-forward parameter."""

    name: str
    kind: inspect._ParameterKind
    default: object = _REQUIRED


_POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
_KEYWORD_REST = inspect.Parameter.VAR_KEYWORD
_ANIMA_FORWARD = (
    _ParameterContract("x", _POSITIONAL),
    _ParameterContract("timesteps", _POSITIONAL),
    _ParameterContract("context", _POSITIONAL),
    _ParameterContract("kwargs", _KEYWORD_REST),
)
_BLOCK_FORWARD = (
    _ParameterContract("x_B_T_H_W_D", _POSITIONAL),
    _ParameterContract("emb_B_T_D", _POSITIONAL),
    _ParameterContract("crossattn_emb", _POSITIONAL),
    _ParameterContract("rope_emb_L_1_1_D", _POSITIONAL, None),
    _ParameterContract("adaln_lora_B_T_3D", _POSITIONAL, None),
    _ParameterContract("extra_per_block_pos_emb", _POSITIONAL, None),
    _ParameterContract("transformer_options", _POSITIONAL, {}),
)
_ATTENTION_FORWARD = (
    _ParameterContract("x", _POSITIONAL),
    _ParameterContract("context", _POSITIONAL, None),
    _ParameterContract("rope_emb", _POSITIONAL, None),
    _ParameterContract("transformer_options", _POSITIONAL, {}),
)
_MLP_FORWARD = (_ParameterContract("x", _POSITIONAL),)


@dataclass(frozen=True)
class AnimaModuleSurfaceIssue:
    """Describe one installed Anima module-surface mismatch."""

    path: str
    reason: str


class AnimaModuleSurfaceError(ValueError):
    """Report every discoverable Anima module-surface mismatch together."""

    def __init__(self, issues: tuple[AnimaModuleSurfaceIssue, ...]) -> None:
        """Create one actionable aggregate pre-patch diagnostic."""

        self.issues = issues
        details = "\n".join(f"- {issue.path}: {issue.reason}" for issue in issues)
        super().__init__("Installed Anima module surface is incompatible:\n" + details)


@dataclass(frozen=True)
class AnimaLoraTargetModule:
    """Bind one canonical regional-LoRA target to its installed module object."""

    target_name: str
    block_index: int
    family: AnimaLoraTargetFamily
    module: nn.Module
    input_features: int
    output_features: int
    forward_signature: str


@dataclass(frozen=True)
class AnimaBlockModuleSurface:
    """Retain the exact installed owners required to patch one diffusion block."""

    block_index: int
    block: Block
    self_attention: Attention
    cross_attention: Attention
    mlp: GPT2FeedForward
    adaln_self_attention: nn.Sequential
    adaln_cross_attention: nn.Sequential
    adaln_mlp: nn.Sequential
    lora_targets: tuple[AnimaLoraTargetModule, ...]


@dataclass(frozen=True)
class AnimaModuleSurface:
    """Expose the fully validated installed Anima diffusion patch surface."""

    diffusion_model: Anima
    blocks: tuple[AnimaBlockModuleSurface, ...]
    lora_targets: tuple[AnimaLoraTargetModule, ...]


class AnimaModuleSurfaceDiscovery:
    """Validate exact installed Anima owners before any object patch is installed."""

    def discover(self, diffusion_model: object) -> AnimaModuleSurface:
        """Return all exact patch targets or reject the complete observed drift."""

        if type(diffusion_model) is not Anima:
            raise AnimaModuleSurfaceError(
                (
                    AnimaModuleSurfaceIssue(
                        "diffusion_model",
                        "expected the exact installed "
                        "comfy.ldm.anima.model.Anima class",
                    ),
                )
            )
        model = cast(Anima, diffusion_model)
        issues: list[AnimaModuleSurfaceIssue] = []
        self._validate_forward(
            "diffusion_model.forward", model.forward, _ANIMA_FORWARD, issues
        )
        self._validate_model_configuration(model, issues)

        raw_blocks = getattr(model, "blocks", None)
        if not isinstance(raw_blocks, nn.ModuleList):
            issues.append(
                AnimaModuleSurfaceIssue(
                    "diffusion_model.blocks",
                    "expected the installed Anima nn.ModuleList block owner",
                )
            )
            raise AnimaModuleSurfaceError(tuple(issues))
        if len(raw_blocks) != ANIMA_BLOCK_COUNT:
            issues.append(
                AnimaModuleSurfaceIssue(
                    "diffusion_model.blocks",
                    f"expected {ANIMA_BLOCK_COUNT} blocks, observed {len(raw_blocks)}",
                )
            )

        discovered_blocks: list[AnimaBlockModuleSurface] = []
        discovered_targets: list[AnimaLoraTargetModule] = []
        for block_index, raw_block in enumerate(raw_blocks):
            block_surface = self._discover_block(block_index, raw_block, issues)
            if block_surface is None:
                continue
            discovered_blocks.append(block_surface)
            discovered_targets.extend(block_surface.lora_targets)

        self._validate_target_inventory(model, discovered_targets, issues)
        if issues:
            raise AnimaModuleSurfaceError(tuple(issues))
        expected_target_count = ANIMA_BLOCK_COUNT * len(AnimaLoraTargetFamily)
        if len(discovered_targets) != expected_target_count:
            raise AnimaModuleSurfaceError(
                (
                    AnimaModuleSurfaceIssue(
                        "diffusion_model.blocks",
                        f"expected {expected_target_count} LoRA targets, "
                        f"discovered {len(discovered_targets)}",
                    ),
                )
            )
        return AnimaModuleSurface(
            diffusion_model=model,
            blocks=tuple(discovered_blocks),
            lora_targets=tuple(discovered_targets),
        )

    @staticmethod
    def _validate_target_inventory(
        model: Anima,
        targets: list[AnimaLoraTargetModule],
        issues: list[AnimaModuleSurfaceIssue],
    ) -> None:
        """Require canonical installed names and one distinct object per target."""

        named_modules = dict(model.named_modules(remove_duplicate=False))
        first_path_by_identity: dict[int, str] = {}
        for target in targets:
            relative_name = target.target_name.removeprefix("diffusion_model.")
            if named_modules.get(relative_name) is not target.module:
                issues.append(
                    AnimaModuleSurfaceIssue(
                        target.target_name,
                        "installed named-module inventory does not retain "
                        "this exact path",
                    )
                )
            identity = id(target.module)
            first_path = first_path_by_identity.setdefault(identity, target.target_name)
            if first_path != target.target_name:
                issues.append(
                    AnimaModuleSurfaceIssue(
                        target.target_name,
                        f"aliases the distinct canonical target {first_path}",
                    )
                )

    @staticmethod
    def _validate_model_configuration(
        model: Anima,
        issues: list[AnimaModuleSurfaceIssue],
    ) -> None:
        """Require the supported Anima diffusion dimensions and AdaLN-LoRA mode."""

        expected_values = {
            "model_channels": 2048,
            "num_blocks": ANIMA_BLOCK_COUNT,
            "num_heads": 16,
            "adaln_lora_dim": 256,
            "use_adaln_lora": True,
        }
        for name, expected in expected_values.items():
            observed = getattr(model, name, None)
            if observed != expected or type(observed) is not type(expected):
                issues.append(
                    AnimaModuleSurfaceIssue(
                        f"diffusion_model.{name}",
                        f"expected {expected!r}, observed {observed!r}",
                    )
                )

    def _discover_block(
        self,
        block_index: int,
        raw_block: nn.Module,
        issues: list[AnimaModuleSurfaceIssue],
    ) -> AnimaBlockModuleSurface | None:
        """Validate and describe one exact installed Cosmos diffusion block."""

        block_path = f"diffusion_model.blocks.{block_index}"
        if type(raw_block) is not Block:
            issues.append(
                AnimaModuleSurfaceIssue(
                    block_path,
                    "expected the exact installed "
                    "comfy.ldm.cosmos.predict2.Block class",
                )
            )
            return None
        block = raw_block
        self._validate_forward(
            f"{block_path}.forward", block.forward, _BLOCK_FORWARD, issues
        )
        self_attention = self._exact_owner(
            block,
            "self_attn",
            Attention,
            block_path,
            issues,
        )
        cross_attention = self._exact_owner(
            block,
            "cross_attn",
            Attention,
            block_path,
            issues,
        )
        mlp = self._exact_owner(
            block,
            "mlp",
            GPT2FeedForward,
            block_path,
            issues,
        )
        adaln_self = self._adaln_owner(
            block, "adaln_modulation_self_attn", block_path, issues
        )
        adaln_cross = self._adaln_owner(
            block, "adaln_modulation_cross_attn", block_path, issues
        )
        adaln_mlp = self._adaln_owner(block, "adaln_modulation_mlp", block_path, issues)
        if self_attention is not None:
            self._validate_forward(
                f"{block_path}.self_attn.forward",
                self_attention.forward,
                _ATTENTION_FORWARD,
                issues,
            )
        if cross_attention is not None:
            self._validate_forward(
                f"{block_path}.cross_attn.forward",
                cross_attention.forward,
                _ATTENTION_FORWARD,
                issues,
            )
        if mlp is not None:
            self._validate_forward(
                f"{block_path}.mlp.forward", mlp.forward, _MLP_FORWARD, issues
            )

        targets = self._discover_targets(block_index, block, issues)
        if (
            self_attention is None
            or cross_attention is None
            or mlp is None
            or adaln_self is None
            or adaln_cross is None
            or adaln_mlp is None
        ):
            return None
        return AnimaBlockModuleSurface(
            block_index=block_index,
            block=block,
            self_attention=self_attention,
            cross_attention=cross_attention,
            mlp=mlp,
            adaln_self_attention=adaln_self,
            adaln_cross_attention=adaln_cross,
            adaln_mlp=adaln_mlp,
            lora_targets=targets,
        )

    @staticmethod
    def _exact_owner(
        block: Block,
        attribute: str,
        expected_type: type[_OwnerType],
        block_path: str,
        issues: list[AnimaModuleSurfaceIssue],
    ) -> _OwnerType | None:
        """Return one exact installed owner or append its type mismatch."""

        owner = getattr(block, attribute, None)
        if type(owner) is not expected_type:
            issues.append(
                AnimaModuleSurfaceIssue(
                    f"{block_path}.{attribute}",
                    "expected exact "
                    f"{expected_type.__module__}.{expected_type.__name__}",
                )
            )
            return None
        return owner

    @staticmethod
    def _adaln_owner(
        block: Block,
        attribute: str,
        block_path: str,
        issues: list[AnimaModuleSurfaceIssue],
    ) -> nn.Sequential | None:
        """Return one exact three-stage AdaLN-LoRA owner or record its mismatch."""

        owner = getattr(block, attribute, None)
        path = f"{block_path}.{attribute}"
        if type(owner) is not nn.Sequential:
            issues.append(
                AnimaModuleSurfaceIssue(path, "expected exact torch.nn.Sequential")
            )
            return None
        if len(owner) != 3 or type(owner[0]) is not nn.SiLU:
            issues.append(
                AnimaModuleSurfaceIssue(
                    path,
                    "expected SiLU followed by the two AdaLN-LoRA linear stages",
                )
            )
            return None
        return owner

    def _discover_targets(
        self,
        block_index: int,
        block: Block,
        issues: list[AnimaModuleSurfaceIssue],
    ) -> tuple[AnimaLoraTargetModule, ...]:
        """Resolve all 16 canonical target modules from one block by identity."""

        targets: list[AnimaLoraTargetModule] = []
        for family in AnimaLoraTargetFamily:
            target_name = anima_lora_target_name(block_index, family)
            target = self._resolve_module(block, family.value)
            if target is None:
                issues.append(
                    AnimaModuleSurfaceIssue(
                        target_name,
                        "canonical target path does not resolve to a module",
                    )
                )
                continue
            expected_input, expected_output = expected_anima_lora_features(family)
            observed_input = getattr(target, "in_features", None)
            observed_output = getattr(target, "out_features", None)
            if observed_input != expected_input or observed_output != expected_output:
                issues.append(
                    AnimaModuleSurfaceIssue(
                        target_name,
                        "expected linear features "
                        f"input={expected_input}, output={expected_output}; observed "
                        f"input={observed_input!r}, output={observed_output!r}",
                    )
                )
                continue
            forward = getattr(target, "forward", None)
            signature = self._single_input_signature(target_name, forward, issues)
            if signature is None:
                continue
            targets.append(
                AnimaLoraTargetModule(
                    target_name=target_name,
                    block_index=block_index,
                    family=family,
                    module=target,
                    input_features=expected_input,
                    output_features=expected_output,
                    forward_signature=signature,
                )
            )
        return tuple(targets)

    @staticmethod
    def _resolve_module(root: nn.Module, relative_path: str) -> nn.Module | None:
        """Resolve one dotted module path without mutating the installed graph."""

        current: object = root
        for part in relative_path.split("."):
            if part.isdecimal() and isinstance(current, (nn.Sequential, nn.ModuleList)):
                index = int(part)
                if not 0 <= index < len(current):
                    return None
                current = current[index]
            else:
                current = getattr(current, part, None)
            if current is None:
                return None
        return current if isinstance(current, nn.Module) else None

    @staticmethod
    def _single_input_signature(
        path: str,
        forward: object,
        issues: list[AnimaModuleSurfaceIssue],
    ) -> str | None:
        """Require a target forward callable that accepts exactly one tensor input."""

        if not callable(forward):
            issues.append(
                AnimaModuleSurfaceIssue(path, "module forward is not callable")
            )
            return None
        try:
            signature = inspect.signature(forward)
            signature.bind(object())
        except (TypeError, ValueError) as error:
            issues.append(
                AnimaModuleSurfaceIssue(
                    f"{path}.forward",
                    f"must accept one positional tensor input: {error}",
                )
            )
            return None
        return str(signature)

    @staticmethod
    def _validate_forward(
        path: str,
        forward: Callable[..., object],
        expected: tuple[_ParameterContract, ...],
        issues: list[AnimaModuleSurfaceIssue],
    ) -> None:
        """Compare one bound forward callable with its exact installed contract."""

        try:
            observed = tuple(inspect.signature(forward).parameters.values())
        except (TypeError, ValueError) as error:
            issues.append(
                AnimaModuleSurfaceIssue(
                    path, f"cannot inspect forward signature: {error}"
                )
            )
            return
        if len(observed) != len(expected):
            issues.append(
                AnimaModuleSurfaceIssue(
                    path,
                    f"expected {len(expected)} parameters, observed {len(observed)}",
                )
            )
            return
        mismatches: list[str] = []
        for actual, contract in zip(observed, expected, strict=True):
            if actual.name != contract.name or actual.kind is not contract.kind:
                mismatches.append(
                    f"expected {contract.name}:{contract.kind.name}, observed "
                    f"{actual.name}:{actual.kind.name}"
                )
                continue
            if contract.default is _REQUIRED:
                if actual.default is not inspect.Parameter.empty:
                    mismatches.append(f"{contract.name} must be required")
            elif (
                actual.default is inspect.Parameter.empty
                or actual.default != contract.default
            ):
                mismatches.append(
                    f"{contract.name} default must be {contract.default!r}"
                )
        if mismatches:
            issues.append(AnimaModuleSurfaceIssue(path, "; ".join(mismatches)))


ANIMA_MODULE_SURFACE_DISCOVERY = AnimaModuleSurfaceDiscovery()
