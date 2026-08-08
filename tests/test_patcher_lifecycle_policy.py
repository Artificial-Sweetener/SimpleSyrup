# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Enforce the single first-party Comfy patcher lifecycle boundary."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "simple_syrup"
LIFECYCLE_MODULE = "simple_syrup/runtime/patcher_lifecycle.py"
FORBIDDEN_PATCHER_CALLS = frozenset(
    {
        "add_object_patch",
        "add_patches",
        "clip_layer",
        "register_all_hook_patches",
        "set_tokenizer_option",
    }
)
FORBIDDEN_PATCHER_WRITES = frozenset({"forced_hooks", "use_clip_schedule"})
APPROVED_VALUE_CLONES = Counter(
    {
        ("simple_syrup/domain/segs_tiled_diffusion.py", "mask"): 2,
        ("simple_syrup/image/crop_composite.py", "image"): 1,
        (
            "simple_syrup/image/resize_service.py",
            "values.expand(cropped.shape[0], cropped.shape[1], "
            "plan.output_height, plan.output_width)",
        ): 1,
        (
            "simple_syrup/masking/prompt_segs_with_sam_service.py",
            "crop_image(image_tensor, crop_region).detach()",
        ): 1,
        (
            "simple_syrup/masking/prompt_segs_with_sam_service.py",
            "crop_mask(final_mask, crop_region).detach()",
        ): 1,
        ("simple_syrup/runtime/sam_region_overlay_renderer.py", "image.detach()"): 2,
        (
            "simple_syrup/services/detail_segs_as_regions_service.py",
            "image_tensor",
        ): 1,
        (
            "simple_syrup/services/detail_segs_by_scale_factor_service.py",
            "image_tensor",
        ): 2,
        (
            "simple_syrup/services/detail_segs_by_scale_factor_tiled_diffusion_service.py",
            "image_tensor",
        ): 2,
        (
            "simple_syrup/services/mask_to_segs_service.py",
            "crop_image(image_tensor, crop_region).detach()",
        ): 1,
        (
            "simple_syrup/services/mask_to_segs_service.py",
            "cropped_segment_mask",
        ): 1,
        (
            "simple_syrup/services/segs_detection_service.py",
            "crop_mask(mask, crop_region).detach()",
        ): 1,
        (
            "simple_syrup/services/segs_detection_service.py",
            "crop_image(image_tensor, crop_region).detach()",
        ): 1,
        (
            "simple_syrup/services/segs_from_sam_output_service.py",
            "image[:, crop_region.top:crop_region.bottom, "
            "crop_region.left:crop_region.right, :].detach()",
        ): 1,
        (
            "simple_syrup/services/segs_from_sam_output_service.py",
            "local_mask.unsqueeze(0).detach()",
        ): 1,
        (
            "simple_syrup/services/segs_output_service.py",
            "crop_mask(combined_mask, crop_region).detach()",
        ): 1,
        (
            "simple_syrup/services/segs_output_service.py",
            "crop_image(image_tensor, crop_region).detach()",
        ): 1,
        (
            "simple_syrup/services/simple_preview_segs_service.py",
            "image.detach().cpu()",
        ): 1,
    }
)


def test_first_party_patcher_lifecycle_has_one_authoritative_owner() -> None:
    """Reject every first-party clone or patcher mutation outside the owner."""

    observed_clones: Counter[tuple[str, str]] = Counter()
    violations: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        if "third_party" in path.parts or relative_path == LIFECYCLE_MODULE:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        file_violations, clones = _lifecycle_bypasses(tree, relative_path)
        violations.extend(file_violations)
        observed_clones.update(clones)

    unexpected_clones = observed_clones - APPROVED_VALUE_CLONES
    missing_clones = APPROVED_VALUE_CLONES - observed_clones
    assert violations == []
    assert unexpected_clones == Counter()
    assert missing_clones == Counter()


def test_policy_detects_a_new_direct_patcher_code_path() -> None:
    """Prove the policy rejects clone, mutation, and object-patch bypasses."""

    source = """
def unsafe(model):
    cloned = model.clone()
    cloned.set_model_unet_function_wrapper(lambda args: args)
    cloned.add_object_patch("encode", model.encode)
    cloned.forced_hooks = object()
    return cloned
"""
    violations, clones = _lifecycle_bypasses(
        ast.parse(source),
        "simple_syrup/runtime/new_feature.py",
    )

    assert len(violations) == 3
    assert clones == Counter({("simple_syrup/runtime/new_feature.py", "model"): 1})


def _lifecycle_bypasses(
    tree: ast.AST,
    relative_path: str,
) -> tuple[list[str], Counter[tuple[str, str]]]:
    """Return direct patcher mutations and all clone callsites in one tree."""

    violations: list[str] = []
    clones: Counter[tuple[str, str]] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr == "clone":
                clones[(relative_path, ast.unparse(node.value))] += 1
            elif _is_forbidden_patcher_call(node.attr):
                violations.append(f"{relative_path}:{node.lineno}: {node.attr}")
        elif isinstance(node, ast.Call) and _uses_forbidden_dynamic_attribute(node):
            violations.append(f"{relative_path}:{node.lineno}: dynamic patcher access")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for target in _assignment_targets(node):
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in FORBIDDEN_PATCHER_WRITES
                ):
                    violations.append(
                        f"{relative_path}:{node.lineno}: write {target.attr}"
                    )
    return violations, clones


def _uses_forbidden_dynamic_attribute(node: ast.Call) -> bool:
    """Return whether getattr or setattr hides a protected patcher attribute."""

    if not isinstance(node.func, ast.Name) or node.func.id not in {
        "getattr",
        "setattr",
    }:
        return False
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
        return False
    attribute_name = node.args[1].value
    return isinstance(attribute_name, str) and (
        _is_forbidden_patcher_call(attribute_name)
        or attribute_name in FORBIDDEN_PATCHER_WRITES
    )


def _is_forbidden_patcher_call(attribute_name: str) -> bool:
    """Return whether an attribute mutates a managed MODEL or CLIP value."""

    return (
        attribute_name.startswith("set_model_")
        or attribute_name in FORBIDDEN_PATCHER_CALLS
    )


def _assignment_targets(
    node: ast.Assign | ast.AnnAssign | ast.AugAssign,
) -> tuple[ast.expr, ...]:
    """Normalize assignment node targets for lifecycle-policy inspection."""

    if isinstance(node, ast.Assign):
        return tuple(node.targets)
    return (node.target,)
