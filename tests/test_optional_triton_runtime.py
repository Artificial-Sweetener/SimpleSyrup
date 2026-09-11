# SimpleSyrup - workflow-focused ComfyUI extensions for image generation
# Copyright (C) 2026  Artificial Sweetener and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove Triton remains an optional CUDA acceleration dependency."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from simple_syrup.runtime.regional_lora.triton_runtime import (
    TritonRuntimeResolver,
)


def test_node_registration_and_cpu_accumulation_do_not_import_triton() -> None:
    """Load public nodes and execute CPU accumulation with Triton blocked."""

    script = """
import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parents[1]))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()

class BlockTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("blocked optional Triton", name=fullname)
        return None

sys.meta_path.insert(0, BlockTriton())
import torch
from simple_syrup.nodes_v3 import get_nodes
from simple_syrup.runtime.regional_lora import ordered_accumulation

assert get_nodes()
base = torch.tensor([1.0, 2.0])
result = ordered_accumulation.OrderedTensorAccumulator().accumulate(
    base,
    (torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])),
)
assert result.tolist() == [9.0, 12.0]
assert not any(name == "triton" or name.startswith("triton.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_resolver_caches_one_missing_result_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Treat an absent Triton package as one observable optional miss."""

    calls: list[str] = []

    def missing_import(name: str) -> object:
        calls.append(name)
        raise ModuleNotFoundError("missing", name="triton")

    resolver = TritonRuntimeResolver(import_module=missing_import)

    with caplog.at_level("WARNING"):
        assert resolver.resolve("fake.backend") is None
        assert resolver.resolve("fake.backend") is None

    assert calls == ["fake.backend"]
    assert [record.message for record in caplog.records] == [
        "Triton acceleration is unavailable; using the Torch execution path"
    ]


def test_resolver_exposes_broken_backend_import_with_original_cause() -> None:
    """Fail visibly when a present backend cannot initialize correctly."""

    failure = RuntimeError("JIT initialization failed")

    def broken_import(_name: str) -> object:
        raise failure

    resolver = TritonRuntimeResolver(import_module=broken_import)

    with pytest.raises(RuntimeError, match="failed to initialize") as raised:
        resolver.resolve("fake.backend")

    assert raised.value.__cause__ is failure


def test_resolver_is_thread_safe_and_returns_one_cached_backend() -> None:
    """Publish exactly one imported backend across concurrent callers."""

    from concurrent.futures import ThreadPoolExecutor

    backend = object()
    calls: list[str] = []

    def import_backend(name: str) -> object:
        calls.append(name)
        return backend

    resolver = TritonRuntimeResolver(import_module=import_backend)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(resolver.resolve, ("fake.backend",) * 32))

    assert all(result is backend for result in results)
    assert calls == ["fake.backend"]


@pytest.mark.parametrize(
    "missing_name",
    ["fake.backend", "unrelated_dependency"],
)
def test_resolver_does_not_hide_non_triton_module_failures(missing_name: str) -> None:
    """Reserve optional fallback exclusively for the Triton package family."""

    def missing_import(_name: str) -> object:
        raise ModuleNotFoundError("missing", name=missing_name)

    resolver = TritonRuntimeResolver(
        import_module=missing_import,
    )

    with pytest.raises(RuntimeError, match="failed to initialize"):
        resolver.resolve("fake.backend")
