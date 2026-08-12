"""Prove exact benchmark-only regional HookGroup rejection fixtures."""

from __future__ import annotations

import asyncio

import comfy.hooks
import pytest
import torch

from tools.attention_coupling_benchmark.comfy_probe import comfy_entrypoint
from tools.attention_coupling_benchmark.comfy_probe.regional_hook_fixture import (
    INCOMPLETE_STANDARD_PAIR,
    LLM_ADAPTER_TARGET,
    MODEL_AS_LORA_FORMAT,
    NON_DIFFUSION_OWNER_BUNDLE,
    NON_WEIGHT_HOOK,
    REGIONAL_HOOK_FIXTURES,
    UNSUPPORTED_ANIMA_TARGET,
    UNSUPPORTED_SUFFIX_FORMAT,
    CreateRegionalHookFixtureV3,
    create_regional_hook_fixture,
)


def test_fixture_node_is_a_closed_dev_only_hooks_source() -> None:
    """Keep the managed rejection seam out of product node exports."""

    schema = CreateRegionalHookFixtureV3.define_schema()

    assert schema.node_id == "SimpleSyrupBenchmark.CreateRegionalHookFixture"
    assert schema.is_dev_only is True
    assert schema.inputs[0].options == list(REGIONAL_HOOK_FIXTURES)
    assert schema.outputs[0].io_type == "HOOKS"


def test_fixture_node_is_registered_only_by_the_benchmark_extension() -> None:
    """Expose the fixture to managed evidence without adding a product export."""

    extension = asyncio.run(comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())

    assert CreateRegionalHookFixtureV3 in nodes


def test_non_weight_fixture_contains_only_an_installed_transformer_hook() -> None:
    """Construct the unsupported hook class without a model payload."""

    hooks = create_regional_hook_fixture(NON_WEIGHT_HOOK)

    assert len(hooks.hooks) == 1
    assert isinstance(hooks.hooks[0], comfy.hooks.TransformerOptionsHook)
    assert hooks.get_type(comfy.hooks.EnumHookType.Weight) == []


def test_converted_model_fixture_retains_installed_patch_tuple_shape() -> None:
    """Distinguish Comfy model-as-LoRA patches from raw standard pairs."""

    hook = _weight_hook(MODEL_AS_LORA_FORMAT)

    assert hook.need_weight_init is False
    assert tuple(hook.weights) == ("diffusion_model.blocks.0.self_attn.q_proj.weight",)
    patch = hook.weights["diffusion_model.blocks.0.self_attn.q_proj.weight"]
    assert patch[0] == "model_as_lora"
    assert isinstance(patch[1][0], torch.Tensor)


@pytest.mark.parametrize(
    ("fixture", "expected_keys"),
    [
        (
            UNSUPPORTED_SUFFIX_FORMAT,
            {
                "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight",
                "diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight",
            },
        ),
        (
            INCOMPLETE_STANDARD_PAIR,
            {"diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight"},
        ),
        (
            UNSUPPORTED_ANIMA_TARGET,
            {
                "diffusion_model.blocks.0.unknown_projection.lora_A.weight",
                "diffusion_model.blocks.0.unknown_projection.lora_B.weight",
            },
        ),
        (
            LLM_ADAPTER_TARGET,
            {
                "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_A.weight",
                "diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.lora_B.weight",
            },
        ),
        (
            NON_DIFFUSION_OWNER_BUNDLE,
            {
                "lora_te_model.layers.0.self_attn.q_proj.lora_A.weight",
                "lora_te_model.layers.0.self_attn.q_proj.lora_B.weight",
                "vae.decoder.conv_in.lora_A.weight",
                "vae.decoder.conv_in.lora_B.weight",
            },
        ),
    ],
)
def test_raw_fixtures_preserve_exact_cpu_tensor_mappings(
    fixture: str,
    expected_keys: set[str],
) -> None:
    """Keep format and target classification in production admission owners."""

    hook = _weight_hook(fixture)

    assert hook.need_weight_init is True
    assert set(hook.weights) == expected_keys
    assert all(value.device.type == "cpu" for value in hook.weights.values())


def test_fixture_factory_rejects_unknown_values() -> None:
    """Prevent benchmark graphs from inventing undeclared hook shapes."""

    with pytest.raises(ValueError, match="Unknown regional hook fixture"):
        create_regional_hook_fixture("unknown")


def _weight_hook(fixture: str) -> comfy.hooks.WeightHook:
    """Return the fixture's one exact installed WeightHook."""

    hooks = create_regional_hook_fixture(fixture)
    weight_hooks = hooks.get_type(comfy.hooks.EnumHookType.Weight)
    assert len(weight_hooks) == 1
    hook = weight_hooks[0]
    assert isinstance(hook, comfy.hooks.WeightHook)
    return hook
