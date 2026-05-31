# Clean V3-Only Node Export Plan

## Goal

Move SimpleSyrup to a clean Comfy v3-only node export model while preserving the current SimpleSyrup node set through v3 schemas.

After this work is complete:

- `comfy_entrypoint()` is the only supported ComfyUI node registration surface.
- `simple_syrup/nodes_v3/__init__.py::get_nodes()` is the authoritative node list.
- The root package no longer exposes `NODE_CLASS_MAPPINGS` or `NODE_DISPLAY_NAME_MAPPINGS`.
- Legacy Comfy mapping exports are removed instead of maintained as a migration path.
- Every SimpleSyrup node that should remain available in ComfyUI is exported as a v3 node.
- Existing workflow-facing node ids such as `SimpleSyrup.BatchRegionConditioning` remain stable where the same node continues to exist.

## Decisions Already Made

- This is a clean break.
- Older ComfyUI installs that only support legacy `NODE_CLASS_MAPPINGS` are not supported.
- Do not build a migration path, compatibility shim, fallback registry, dual export path, or transitional adapter for legacy Comfy exports.
- Do not keep root-level `NODE_CLASS_MAPPINGS` or `NODE_DISPLAY_NAME_MAPPINGS` for third-party import compatibility.
- Export every maintained node through Comfy v3 only.
- Keep stable node ids for maintained nodes so v3-capable Comfy can resolve the same `SimpleSyrup.*` identifiers.
- Existing `simple_syrup.nodes.*` modules may temporarily remain as implementation modules when a v3 wrapper delegates to them, but they must not own registration.
- Long-term architecture should move behavior out of legacy node classes and into services/domain/runtime modules, with v3 wrappers acting as the only Comfy-facing node API layer.
- Prompt Control-dependent nodes remain conditional, but the conditional export decision lives only in `simple_syrup/nodes_v3/__init__.py`.

## Implementation Status

- [x] Phase 1: Characterize current export intent.
  - Landing note: compared the legacy mapping registry against `nodes_v3.get_nodes()` and used that inventory to drive the wrapper list below.
- [x] Phase 2: Add missing v3 wrappers.
  - Landing note: added explicit v3 wrapper classes for legacy-only nodes in `simple_syrup/nodes_v3/legacy_node_wrappers.py`; they delegate to existing implementation classes without preserving legacy registration.
- [x] Phase 3: Make the v3 registry complete.
  - Landing note: `simple_syrup/nodes_v3/__init__.py::get_nodes()` now returns every maintained base node plus Prompt Control nodes only when available.
- [x] Phase 4: Remove legacy root exports.
  - Landing note: root `__init__.py` no longer imports, exposes, or exports legacy mapping dictionaries.
- [x] Phase 5: Remove legacy registry authority.
  - Landing note: `simple_syrup/nodes/__init__.py` is now a package marker for implementation modules, not a ComfyUI registry.
- [x] Phase 6: Update tests.
  - Landing note: registration and tooltip tests now assert the v3-only contract; added adapter coverage for schema conversion and hidden-input delegation.
- [x] Phase 7: Update documentation and guardrails.
  - Landing note: `AGENTS.md` now identifies Comfy v3 as the only supported export path and removes dual-export requirements.
- [x] Final verification gates.
  - Landing note: `ruff format .`, `ruff check .`, `mypy --strict simple_syrup tests`, and full `pytest -n auto -q` all pass. The remaining production `NODE_CLASS_MAPPINGS` text is a runtime read of ComfyUI's global registry in latent provenance tracing, not a SimpleSyrup registration/export path.

## Non-Goals

- Do not support old ComfyUI installations that do not call `comfy_entrypoint()`.
- Do not preserve imports of `SimpleSyrup.NODE_CLASS_MAPPINGS`.
- Do not preserve imports of `SimpleSyrup.NODE_DISPLAY_NAME_MAPPINGS`.
- Do not keep `simple_syrup/nodes/__init__.py` as a registry.
- Do not rename maintained node ids as part of this change.
- Do not use this refactor to change node behavior, defaults, output order, return shapes, categories, or display names unless a v3 schema cannot represent the current contract and the behavior change is explicitly tested.

## Current Export State

The repository currently has two export paths:

- Legacy mappings in `simple_syrup/nodes/__init__.py`.
- Comfy v3 node list in `simple_syrup/nodes_v3/__init__.py`.

The root `__init__.py` currently imports and exports legacy mappings while also defining `comfy_entrypoint()`. The final state must remove the legacy mapping import/export from the root package.

## Existing V3 Coverage

The following nodes already have v3 wrappers or v3 classes and must remain in the v3 registry:

- `SimpleSyrup.BatchRegionConditioning`
- `SimpleSyrup.BatchSEGS`
- `SimpleSyrup.ExternalLLMPrompt`
- `SimpleSyrup.ScaleFactor`
- `SimpleSyrup.SimpleLoadCheckpoint`
- `SimpleSyrup.TagSEGSWithWD14`
- `SimpleSyrup.TileAndTagSEGS`
- `SimpleSyrup.VAEDecodeOptions`
- `SimpleSyrup.VAEEncodeOptions`
- `SimpleSyrup.WD14TaggerLoader`
- `SimpleSyrup.EncodePromptBatchWithPromptControl`, when Prompt Control is available
- `SimpleSyrup.ScheduleAndEncodePromptsWithPromptControl`, when Prompt Control is available

Verify this list against the code before implementation. If new v3 wrappers have been added since this plan was written, treat the code as authoritative and update the implementation checklist accordingly.

## Legacy-Only Nodes To Port

The following current legacy mapping nodes must receive v3 wrappers before legacy exports are removed:

- `SimpleSyrup.ConditioningBatchStart`
- `SimpleSyrup.ConditioningBatchAppend`
- `SimpleSyrup.GroundedSAMModelInfo`
- `SimpleSyrup.GroundingDINOModelLoader`
- `SimpleSyrup.KSamplerExtras`
- `SimpleSyrup.KSamplerTiledDiffusion`
- `SimpleSyrup.LayerStyleSAMModelsAdapter`
- `SimpleSyrup.LatentDiagnostics`
- `SimpleSyrup.PromptEncodeStyle`
- `SimpleSyrup.PromptEncodeStyleAndNormalization`
- `SimpleSyrup.PromptSEGSWithSAM`
- `SimpleSyrup.SimpleVAEEncode`
- `SimpleSyrup.UpscaleLatentFromImage`
- `SimpleSyrup.ResizeImageToTarget`
- `SimpleSyrup.DetailSEGSAsRegions`
- `SimpleSyrup.DetailSEGSByScaleFactor`
- `SimpleSyrup.DetailSEGSByScaleFactorTiledDiffusion`
- `SimpleSyrup.SAMModelLoader`
- `SimpleSyrup.Seed`
- `SimpleSyrup.SimpleLoadAnima`
- `SimpleSyrup.LoadUltralyticsModel`
- `SimpleSyrup.DetectSEGSWithUltralytics`
- `SimpleSyrup.EncodePromptBatch`
- `SimpleSyrup.ViTMatteModelLoader`

Port these as maintained v3 nodes. Do not remove a node from the product by omission.

## Final Architecture

### Root Package

File: `__init__.py`

The root package should:

- Set up the local `simple_syrup` package alias if it is still required.
- Define `WEB_DIRECTORY`.
- Define `comfy_entrypoint()`.
- Register backend routes that are required at import time.
- Export `WEB_DIRECTORY` and `comfy_entrypoint` in `__all__`.

The root package should not:

- Import `simple_syrup.nodes.NODE_CLASS_MAPPINGS`.
- Import `simple_syrup.nodes.NODE_DISPLAY_NAME_MAPPINGS`.
- Define `NODE_CLASS_MAPPINGS`.
- Define `NODE_DISPLAY_NAME_MAPPINGS`.
- Include either mapping name in `__all__`.

### V3 Registry

File: `simple_syrup/nodes_v3/__init__.py`

`get_nodes()` should:

- Import v3 node classes lazily inside the function.
- Return every maintained SimpleSyrup v3 node class.
- Keep Prompt Control-only imports inside the availability branch.
- Keep unrelated nodes exported when Prompt Control is unavailable.
- Be the only registry used by ComfyUI.

The module should not:

- Import or depend on legacy mapping dictionaries.
- Mirror any legacy registration structure.
- Use compatibility fallback behavior for legacy ComfyUI.

### V3 Wrappers

Directory: `simple_syrup/nodes_v3/`

Each v3 wrapper should:

- Use the same `node_id` as the maintained legacy node id.
- Use the same display name.
- Use the same category.
- Preserve input names, default values, accepted types, optional inputs, hidden inputs, and widget behavior where v3 supports them.
- Preserve output ids, output types, and return shapes.
- Include concise user-facing descriptions and tooltips.
- Delegate non-trivial behavior to existing services/domain/runtime code.
- Avoid duplicating business logic that already has an authoritative service.
- Narrow `Any` at Comfy boundary points before core logic relies on values.
- Use docstrings for new or changed modules, classes, functions, and methods.

Temporary delegation to a legacy node implementation class is acceptable only as an implementation detail during this clean export break. It must not preserve or reintroduce legacy registration.

### Legacy Node Package

File: `simple_syrup/nodes/__init__.py`

This file must stop being a registry.

Acceptable end states:

- Delete the legacy registry file if no internal imports require `simple_syrup.nodes` package exports.
- Reduce it to a normal package marker with a short module docstring and no node mapping dictionaries.
- Keep selective non-registry imports only if they are required by internal code and do not create an alternate registration surface.

Unacceptable end states:

- `NODE_CLASS_MAPPINGS` still exists.
- `NODE_DISPLAY_NAME_MAPPINGS` still exists.
- `__all__` advertises legacy node mappings.
- The file remains the source of truth for exported nodes.

## Implementation Phases

### Phase 1: Characterize Current Export Intent

1. Read `simple_syrup/nodes/__init__.py`.
2. Record every legacy node id, class, and display name.
3. Read `simple_syrup/nodes_v3/__init__.py`.
4. Record every existing v3 node id.
5. Build a one-to-one checklist showing which maintained nodes already have v3 wrappers and which need wrappers.
6. Add or update tests that prove the current intended node set before removing legacy exports.

The checklist in this plan is the starting point, but implementation must verify against the current files because the working tree may have changed.

### Phase 2: Add Missing V3 Wrappers

For each legacy-only maintained node:

1. Create a v3 wrapper module under `simple_syrup/nodes_v3/`.
2. Implement `define_schema()` using `comfy_api.latest.io`.
3. Implement `execute()` or the correct v3 execution hook for the node.
4. Preserve the node id, display name, category, inputs, outputs, defaults, and return shape.
5. Delegate behavior to the existing service/domain/runtime owner where possible.
6. Add focused tests for the wrapper schema and execution behavior.
7. Add tooltip coverage for visible inputs, optional inputs, hidden inputs when supported, and outputs.

Suggested grouping:

- Simple value/helper nodes: `Seed`, `ScaleFactor` parity checks, conditioning batch nodes.
- Model loader and adapter nodes: SAM, GroundingDINO, ViTMatte, Ultralytics, LayerStyle, Anima.
- Image/latent utility nodes: resize, latent diagnostics, VAE provenance nodes.
- Prompt nodes: encode style, style normalization, prompt batch.
- SEGS nodes: prompt SEGS, detect SEGS, detail SEGS variants.
- Sampler nodes: `KSamplerExtras`, `KSamplerTiledDiffusion`.

Sampler and SEGS nodes are higher risk because they touch workflow-facing behavior, hidden inputs, lazy expansion, or runtime side effects. Add characterization tests before changing those wrappers.

### Phase 3: Make The V3 Registry Complete

1. Update `simple_syrup/nodes_v3/__init__.py`.
2. Import every maintained v3 node lazily inside `get_nodes()`.
3. Return all normal nodes in a deterministic order.
4. Keep Prompt Control-only nodes conditional.
5. Add tests for Prompt Control available and unavailable states.
6. Assert that unavailable Prompt Control removes only Prompt Control-specific nodes and does not remove unrelated v3 nodes.

The deterministic order does not need to match the old legacy dictionary order exactly, but it should be stable and easy to review.

### Phase 4: Remove Legacy Root Exports

1. Edit root `__init__.py`.
2. Remove imports of `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
3. Remove those names from `__all__`.
4. Keep `WEB_DIRECTORY`.
5. Keep route registration.
6. Keep `comfy_entrypoint()`.
7. Add tests that assert the root package has no legacy mapping attributes.
8. Add tests that assert `comfy_entrypoint().get_node_list()` returns every maintained node id.

This is the main clean-break step.

### Phase 5: Remove Legacy Registry Authority

1. Edit `simple_syrup/nodes/__init__.py`.
2. Remove `NODE_CLASS_MAPPINGS`.
3. Remove `NODE_DISPLAY_NAME_MAPPINGS`.
4. Remove registry-only imports.
5. Remove registry-only `__all__` entries.
6. Keep the package importable if internal modules still import from `simple_syrup.nodes`.

Do not delete individual legacy node implementation modules in this phase unless their behavior has been fully moved into services/domain/runtime and all callsites have been updated.

### Phase 6: Update Tests

Update the test suite to match the new contract.

Remove or rewrite tests that assert:

- Root package exposes `NODE_CLASS_MAPPINGS`.
- Root package exposes `NODE_DISPLAY_NAME_MAPPINGS`.
- Node ids are registered through `simple_syrup.nodes.NODE_CLASS_MAPPINGS`.
- Display names are registered through `simple_syrup.nodes.NODE_DISPLAY_NAME_MAPPINGS`.

Add or update tests that assert:

- Root package exposes `comfy_entrypoint`.
- Root package exposes `WEB_DIRECTORY`.
- Root package does not expose `NODE_CLASS_MAPPINGS`.
- Root package does not expose `NODE_DISPLAY_NAME_MAPPINGS`.
- `comfy_entrypoint().get_node_list()` includes every maintained node id.
- Every v3 schema has a non-empty `description`.
- Every v3 visible input has a tooltip where the Comfy v3 API supports it.
- Every v3 output has a tooltip where the Comfy v3 API supports it.
- Prompt Control conditional nodes appear only when Prompt Control is available.
- Prompt Control unavailable state does not remove unrelated nodes.
- V3 wrapper execution preserves legacy behavior for each ported node.

Prefer schema tests for Comfy-facing contracts and service/domain tests for behavior.

### Phase 7: Update Documentation And Guardrails

1. Update `AGENTS.md` to say v3 is the only export path.
2. Remove language requiring updates to legacy mapping exports.
3. Remove language requiring the root package to expose legacy mappings.
4. Update node registration testing rules to reference v3 schemas and `comfy_entrypoint()`.
5. Update Definition of Done language so new, renamed, or removed nodes are verified through v3 only.
6. Search docs for stale statements that claim legacy and v3 exports are both required.
7. Update only documentation that directly describes current engineering rules or current user-facing behavior.

Do not create additional design docs unless explicitly requested.

## Testing Requirements

Run focused tests while implementing, then run the full required gates before completion.

Required commands from repository root:

```powershell
..\..\venv\Scripts\ruff.exe format .
..\..\venv\Scripts\ruff.exe check .
..\..\venv\Scripts\mypy.exe --strict simple_syrup tests
..\..\venv\Scripts\python.exe -m pytest -n auto -q
```

If frontend files are touched, also run:

```powershell
npm ci
npm run lint:web
npm run typecheck:web
npm run test:web
npm run build:web
```

Do not use global Python, global pytest, global ruff, global mypy, or a repository-local `.venv`.

## Acceptance Criteria

The work is complete when all of these are true:

- `SimpleSyrup.NODE_CLASS_MAPPINGS` is gone.
- `SimpleSyrup.NODE_DISPLAY_NAME_MAPPINGS` is gone.
- `simple_syrup.nodes.NODE_CLASS_MAPPINGS` is gone.
- `simple_syrup.nodes.NODE_DISPLAY_NAME_MAPPINGS` is gone.
- `comfy_entrypoint()` is present and returns a Comfy extension.
- `comfy_entrypoint().get_node_list()` returns every maintained non-conditional node.
- Prompt Control conditional nodes are exported only when Prompt Control is available.
- Every maintained current node has a v3 schema.
- V3 node ids match maintained current `SimpleSyrup.*` ids.
- Node display names, categories, inputs, defaults, outputs, and execution behavior are preserved unless explicitly changed and tested.
- Tests no longer depend on legacy mapping exports.
- `AGENTS.md` describes v3-only exports.
- Required Python gates pass.
- Required frontend gates pass if frontend files are touched.

## Review Checklist

Before requesting review, verify:

- Search for `NODE_CLASS_MAPPINGS` returns no production registration path.
- Search for `NODE_DISPLAY_NAME_MAPPINGS` returns no production registration path.
- Search for `simple_syrup.nodes_v3` confirms `get_nodes()` is the sole node registry.
- Root `__all__` includes only supported public exports.
- No compatibility layer was added for old ComfyUI.
- No maintained node disappeared from the v3 node list.
- No Prompt Control import happens when Prompt Control is unavailable.
- Error messages exposed to Comfy users remain actionable.
- Tooltips remain concise and user-facing.

## Notes For New Contributors

Comfy v3 nodes are schema-driven. Start by opening an existing wrapper in `simple_syrup/nodes_v3/`, then mirror that pattern for the node you are porting. The wrapper should describe the Comfy-facing contract and delegate real work to existing code.

When in doubt about behavior, write a characterization test around the current legacy node before creating the v3 wrapper. The clean break removes the old export path, not the maintained node behavior.
