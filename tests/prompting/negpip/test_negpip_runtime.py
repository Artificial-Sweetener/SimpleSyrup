# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove family-specific NegPiP tensor and wrapper invariants."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from simple_syrup.runtime.negpip.anima import (
    CONDITION_MASK_KEY as ANIMA_CONDITION_MASK_KEY,
)
from simple_syrup.runtime.negpip.anima import (
    TRANSFORMER_MASK_KEY as ANIMA_TRANSFORMER_MASK_KEY,
)
from simple_syrup.runtime.negpip.anima import (
    anima_attn2_negpip,
    anima_diffusion_negpip_wrapper,
    anima_extra_conds_negpip_wrapper,
)
from simple_syrup.runtime.negpip.krea2 import (
    CONDITION_MASK_KEY as KREA_CONDITION_MASK_KEY,
)
from simple_syrup.runtime.negpip.krea2 import (
    ENCODER_MASK_KEY,
    encode_krea2_token_weights_negpip,
    krea2_attn1_negpip,
    krea2_diffusion_negpip_wrapper,
    krea2_extra_conds_negpip_wrapper,
)
from simple_syrup.runtime.negpip.krea2 import (
    TRANSFORMER_MASK_KEY as KREA_TRANSFORMER_MASK_KEY,
)
from simple_syrup.runtime.negpip.standard import (
    encode_token_weights_negpip,
    standard_attn2_negpip,
)


class _StandardEncoder:
    """Produce deterministic token and empty-prompt embeddings."""

    special_tokens: dict[str, int] = {}

    def gen_empty_tokens(
        self,
        special_tokens: dict[str, int],
        length: int,
    ) -> list[int]:
        """Return a fixed empty token row matching the requested length."""

        del special_tokens
        return [0] * length

    def encode(self, sections: list[list[object]]) -> tuple[torch.Tensor, None]:
        """Map source tokens to scalar embeddings and empty tokens to one."""

        rows = [
            [[1.0 if token == 0 else float(cast(int, token))] for token in section]
            for section in sections
        ]
        return torch.tensor(rows), None


def test_standard_negpip_interleaves_magnitude_keys_and_signed_values() -> None:
    """Standard encoding doubles tokens and signs only the value positions."""

    encoded, pooled = encode_token_weights_negpip(
        cast(Any, _StandardEncoder()),
        [[(3, -2.0), (5, 0.5)]],
    )

    assert pooled is None
    assert isinstance(encoded, torch.Tensor)
    assert encoded.flatten().tolist() == [5.0, -5.0, 3.0, 3.0]

    query = torch.tensor([[[9.0], [8.0]]])
    key = encoded.clone()
    value = encoded.clone()
    prepared_query, prepared_key, prepared_value = standard_attn2_negpip(
        query,
        key,
        value,
        {},
    )
    assert prepared_query is query
    assert prepared_key.flatten().tolist() == [5.0, 3.0]
    assert prepared_value.flatten().tolist() == [-5.0, 3.0]


def test_anima_negpip_preserves_magnitude_and_propagates_value_mask() -> None:
    """Anima moves signs through conditions and changes only attention values."""

    observed_weights: list[torch.Tensor] = []

    def base_extra_conds(**kwargs: object) -> dict[str, object]:
        weights = kwargs["t5xxl_weights"]
        assert isinstance(weights, torch.Tensor)
        observed_weights.append(weights)
        return {"base": "condition"}

    wrapped = anima_extra_conds_negpip_wrapper(base_extra_conds)
    output = wrapped(t5xxl_weights=torch.tensor([-2.0, 0.5, 1.0]))

    assert torch.equal(observed_weights[0], torch.tensor([2.0, 0.5, 1.0]))
    condition = output[ANIMA_CONDITION_MASK_KEY]
    multiplier = cast(Any, condition).cond
    assert multiplier.shape == (1, 512, 1)
    assert multiplier[0, :3, 0].tolist() == [-1.0, 1.0, 1.0]
    assert torch.all(multiplier[0, 3:, 0] == 1.0)

    captured: dict[str, object] = {}

    def executor(*args: object, **kwargs: object) -> str:
        del args
        captured.update(kwargs)
        return "executed"

    context = torch.zeros((1, 512, 4))
    result = anima_diffusion_negpip_wrapper(
        executor,
        object(),
        object(),
        context,
        transformer_options={"existing": True},
        **{ANIMA_CONDITION_MASK_KEY: multiplier},
    )
    assert result == "executed"
    options = cast(dict[str, object], captured["transformer_options"])
    assert options["existing"] is True
    assert torch.equal(
        cast(torch.Tensor, options[ANIMA_TRANSFORMER_MASK_KEY]), multiplier
    )

    query = torch.ones((1, 1, 3, 1))
    key = torch.full_like(query, 2.0)
    value = torch.tensor([[[[3.0], [4.0], [5.0]]]])
    attention = anima_attn2_negpip(
        query,
        key,
        value,
        extra_options={ANIMA_TRANSFORMER_MASK_KEY: multiplier[:, :3]},
    )
    assert attention["q"] is query
    assert attention["k"] is key
    assert cast(torch.Tensor, attention["v"]).flatten().tolist() == [
        -3.0,
        4.0,
        5.0,
    ]


def test_krea2_negpip_preserves_shape_and_signs_only_text_values() -> None:
    """Krea retains layered encoding and leaves Q, K, and image V untouched."""

    tokens: dict[str, list[list[tuple[object, ...]]]] = {
        "qwen3vl_4b": [
            [
                (151644, 1.0),
                (0, 1.0),
                (198, 1.0),
                (151644, 1.0),
                (872, 1.0),
                (198, 1.0),
                (10, -2.0),
                (11, 0.5),
            ]
        ]
    }
    observed: dict[str, object] = {}

    def original(
        prepared: dict[str, list[list[tuple[object, ...]]]],
        *,
        template_end: int,
    ) -> tuple[torch.Tensor, None, dict[str, object]]:
        observed["tokens"] = prepared
        observed["template_end"] = template_end
        return torch.ones((1, 2, 30_720)), None, {"source": True}

    conditioning, pooled, extra = encode_krea2_token_weights_negpip(
        original,
        tokens,
    )

    conditioning_tensor = cast(torch.Tensor, conditioning)
    assert conditioning_tensor.shape == (1, 2, 30_720)
    assert pooled is None
    absolute = cast(
        dict[str, list[list[tuple[object, ...]]]],
        observed["tokens"],
    )
    assert [pair[1] for pair in absolute["qwen3vl_4b"][0][-2:]] == [2.0, 0.5]
    metadata = cast(dict[str, object], extra)
    multiplier = cast(torch.Tensor, metadata[ENCODER_MASK_KEY])
    assert multiplier.flatten().tolist() == [-1.0, 1.0]

    wrapped_extra = krea2_extra_conds_negpip_wrapper(lambda **kwargs: {})
    processed = wrapped_extra(**{ENCODER_MASK_KEY: multiplier})
    condition = processed[KREA_CONDITION_MASK_KEY]
    processed_multiplier = cast(Any, condition).cond

    captured: dict[str, object] = {}

    def executor(*args: object, **kwargs: object) -> str:
        del args
        captured.update(kwargs)
        return "executed"

    assert (
        krea2_diffusion_negpip_wrapper(
            executor,
            transformer_options={"img_slice": [2, 4]},
            **{KREA_CONDITION_MASK_KEY: processed_multiplier},
        )
        == "executed"
    )
    options = cast(dict[str, Any], captured["transformer_options"])
    assert options["img_slice"] == [2, 4]

    positional_capture: dict[str, object] = {}

    def positional_executor(*args: object, **kwargs: object) -> str:
        positional_capture["args"] = args
        positional_capture["kwargs"] = kwargs
        return "positional"

    positional_options = {"img_slice": [2, 4]}
    assert (
        krea2_diffusion_negpip_wrapper(
            positional_executor,
            object(),
            object(),
            object(),
            None,
            None,
            positional_options,
            **{KREA_CONDITION_MASK_KEY: processed_multiplier},
        )
        == "positional"
    )
    positional_args = cast(tuple[object, ...], positional_capture["args"])
    prepared_positional = cast(dict[str, object], positional_args[5])
    assert prepared_positional is not positional_options
    assert prepared_positional[KREA_TRANSFORMER_MASK_KEY] is processed_multiplier
    assert "transformer_options" not in cast(
        dict[str, object], positional_capture["kwargs"]
    )

    query = torch.arange(8.0).reshape(1, 1, 4, 2)
    key = query + 10.0
    value = query + 20.0
    attention = krea2_attn1_negpip(
        query,
        key,
        value,
        extra_options=options,
    )
    assert attention["q"] is query
    assert attention["k"] is key
    prepared_value = cast(torch.Tensor, attention["v"])
    assert prepared_value[0, 0, 0].tolist() == [-20.0, -21.0]
    assert prepared_value[0, 0, 1].tolist() == [22.0, 23.0]
    assert torch.equal(prepared_value[:, :, 2:], value[:, :, 2:])
    assert torch.equal(value, query + 20.0)


@pytest.mark.parametrize(
    "image_slice",
    (None, [3, 4]),
)
def test_krea2_negpip_rejects_unprovable_text_boundaries(
    image_slice: object,
) -> None:
    """Krea fails closed when the model boundary cannot align to its sign mask."""

    options: dict[str, object] = {KREA_TRANSFORMER_MASK_KEY: torch.ones((1, 2, 1))}
    if image_slice is not None:
        options["img_slice"] = image_slice

    with pytest.raises(ValueError, match="boundary|does not match"):
        krea2_attn1_negpip(
            torch.ones((1, 1, 4, 1)),
            torch.ones((1, 1, 4, 1)),
            torch.ones((1, 1, 4, 1)),
            extra_options=options,
        )
