# Batch SEGS and WD14 SEGS Tagging Plan

## Goal

Add a detector-first regional detailing workflow:

1. Upstream detector nodes create `SEGS` payloads. These can come from multiple Ultralytics detector models, prompted SAM, or any other Impact-compatible SEGS producer.
2. A new **Batch SEGS** node combines those ordered `SEGS` payloads into one ordered `SEGS` payload.
3. A future **Tag SEGS w/ WD14** node runs WD14 over each segment crop and produces a `CONDITIONING_BATCH` aligned to the same segment order.
4. The existing **Detail SEGS as Regions** node receives the batched `SEGS` plus aligned regional conditioning and details all regions in one regional MultiDiffusion pass.

The immediate implementation target is **Batch SEGS**. The WD14 tagging node is included here so the Batch SEGS design supports the intended downstream workflow.

## Implementation Status

- [x] Domain batching policy added in `simple_syrup/domain/segs.py`.
  - Landing note: `batch_segs(...)` returns native immutable SEGS and leaves Impact-compatible conversion to Comfy-facing code.
- [x] Domain tests added in `tests/test_segs_domain.py`.
  - Landing note: tests cover order-preserving flattening, empty SEGS inputs, all-empty output, no-input rejection, and mismatched headers.
- [x] **Batch SEGS** v3 node added in `simple_syrup/nodes_v3/batch_segs.py`.
  - Landing note: the node uses `io.Autogrow.TemplatePrefix` with `segs0` through `segs49`, matching Comfy core batch-node behavior.
- [x] **Batch SEGS** exported through `simple_syrup/nodes_v3/__init__.py`.
  - Landing note: the node is intentionally v3-only and is not registered in legacy mappings.
- [x] V3 node and entrypoint tests added or updated.
  - Landing note: `tests/test_batch_segs_v3_node.py` covers schema and execution; `tests/test_registration.py` covers v3 entrypoint visibility.
- [x] Focused verification passed.
  - Landing note: `..\..\venv\Scripts\python.exe -m pytest -n auto -q tests\test_segs_domain.py tests\test_batch_segs_v3_node.py tests\test_registration.py` passed with 78 tests.
- [x] Format, lint, type check, and full test suite passed for **Batch SEGS**.
  - Landing note: initial Batch SEGS gates passed before later workflow scope additions.
- [x] **Tag SEGS w/ WD14** implemented after scope correction.
  - Landing note: `TagSEGSWithWD14Service` crops existing SEGS in order, runs WD14, prefixes prompts, CLIP-encodes aligned conditioning, and returns the original SEGS with `CONDITIONING_BATCH`.
- [x] **Tag SEGS w/ WD14** exported through legacy mappings and v3 entrypoint.
  - Landing note: unlike **Batch SEGS**, this node is not v3-only because it does not need Autogrow and should be available through every repository-supported export path.
- [x] **Tag SEGS w/ WD14** focused verification passed.
  - Landing note: `..\..\venv\Scripts\python.exe -m pytest -n auto -q tests\test_tag_segs_with_wd14_service.py tests\test_tag_segs_with_wd14_node.py tests\test_tag_segs_with_wd14_v3_node.py tests\test_registration.py tests\test_node_tooltips.py` passed with 57 tests.
- [x] Final gates rerun after **Tag SEGS w/ WD14**.
  - Landing note: gates passed after both SEGS batching and WD14 tagging were implemented.
- [x] **Batch Region Conditioning** implemented after mixed auto/manual conditioning requirement.
  - Landing note: v3-only Autogrow node accepts mixed `CONDITIONING` and `CONDITIONING_BATCH` sockets, flattening existing batches and independent conditionings into one ordered `CONDITIONING_BATCH`.
- [x] Final gates rerun after **Batch Region Conditioning**.
  - Landing note: gates passed after the mixed conditioning batcher was added.
- [x] Legacy-visible **Batch SEGS** and **Batch Region Conditioning** added after Comfy menu visibility check.
  - Landing note: the Autogrow v3 nodes did not appear in the maintainer's current Comfy menu. Legacy two-input chainable nodes were added and registered so both utilities appear through `NODE_CLASS_MAPPINGS`.
- [x] Final gates rerun after legacy-visible batch nodes.
  - Landing note: gates passed after legacy Batch SEGS and Batch Region Conditioning were added.
- [x] Warning handling made explicit before commit.
  - Landing note: pytest now treats unhandled warnings as errors, narrowly filters known third-party SWIG import deprecations, and asserts the intentional PyTorch nested tensor warning in the owning test. Final full suite result: 920 passed with no warning summary.

## Decisions From Maintainer Discussion

- The node name is **Batch SEGS**.
- **Batch SEGS** must be its own node, not folded into **Tile & Tag SEGS**.
- **Tile & Tag SEGS** remains for generated tile SEGS. The new workflow is for existing detector-produced SEGS.
- **Batch SEGS** must support expandable inputs like ComfyUI core nodes such as **Math Expression**, **Batch Images**, **Batch Masks**, and **Batch Latents**.
- Use Comfy v3 `io.Autogrow` for the expandable input UI.
- It is acceptable for **Batch SEGS** to be v3-only. Do not add an awkward fixed-input legacy fallback.
- **Batch SEGS** must accept existing SEGS batches and flatten them in order. If one input contains segments `1 2 3` and another input contains `4 5 6`, the output must contain `1 2 3 4 5 6`.
- Segment order matters because downstream conditioning is index-aligned to segments.
- All inputs must target the same source image dimensions. Reject mismatched `SEGS` headers.

## Existing Code Context

Important files:

- `simple_syrup/domain/segs.py`
  - Owns SEGS coercion, Impact compatibility, sorting, and limiting.
  - Existing functions include `coerce_segs`, `coerce_segs_group`, `to_impact_compatible_segs`, and `to_impact_compatible_segs_group`.
- `simple_syrup/nodes_v3/__init__.py`
  - Exports v3 nodes through `get_nodes()`.
- `simple_syrup/nodes_v3/tile_and_tag_segs.py`
  - Shows the local v3 wrapper pattern.
- `simple_syrup/nodes/__init__.py`
  - Legacy mapping exports. **Batch SEGS** does not need to be added here if implemented as v3-only.
- `simple_syrup/services/tile_and_tag_segs_service.py`
  - Existing WD14 crop-tag-encode orchestration for generated tile SEGS.
- `simple_syrup/nodes/detail_segs_as_regions.py`
  - Existing downstream regional detailer.
  - It consumes `SEGS` plus `CONDITIONING_BATCH`.
- `simple_syrup/services/detail_segs_as_regions_service.py`
  - Validates and pairs region conditioning by segment index.
- `tests/test_segs_domain.py`
  - Existing SEGS domain tests.
- `tests/test_registration.py`
  - Existing registration and v3 entrypoint tests.

Comfy reference examples:

- `E:\ComfyUI\comfy_extras\nodes_math.py`
  - `MathExpressionNode` uses `io.Autogrow.TemplateNames`.
- `E:\ComfyUI\comfy_extras\nodes_post_processing.py`
  - `BatchImagesNode`, `BatchMasksNode`, and `BatchLatentsNode` use `io.Autogrow.TemplatePrefix`.
- `E:\ComfyUI\comfy_api\latest\_io.py`
  - `Autogrow` implementation and dynamic input expansion.

## Architecture Constraints

- Keep Comfy-facing node code thin.
- Put merge policy in the domain layer, not in the node wrapper.
- Domain logic must not import ComfyUI modules.
- Runtime adapters own external system interaction. **Batch SEGS** should not need runtime adapters.
- Preserve public behavior of existing nodes.
- Do not rename existing node IDs, display names, categories, inputs, outputs, or return shapes.
- Do not add internal compatibility shims.
- New or changed code must have docstrings.
- New behavior must be covered by tests.
- Use the ComfyUI virtual environment two directories above the repository for verification.

## Batch SEGS Functional Specification

### Node

Display name:

```text
Batch SEGS
```

Suggested v3 node id:

```text
SimpleSyrup.BatchSEGS
```

Suggested category:

```text
SimpleSyrup/Detection
```

Search aliases:

```text
batch, merge, join, combine, segs
```

Inputs:

- Expandable Autogrow group named `segs_inputs`.
- Template input type: `SEGS`.
- Prefix: `segs`.
- Minimum visible/required inputs: `2`.
- Maximum inputs: `50`, matching Comfy core batch nodes.

Outputs:

- `SEGS`
- Return name: `segs`

### Behavior

Given:

```text
segs0 = ((height, width), [segment1, segment2, segment3])
segs1 = ((height, width), [segment4, segment5, segment6])
```

Return:

```text
((height, width), [segment1, segment2, segment3, segment4, segment5, segment6])
```

Rules:

- Coerce every input using existing SEGS compatibility handling.
- Preserve input order from the Autogrow dict values.
- Preserve order inside each input SEGS payload.
- Reject an empty Autogrow group. This should not happen with `min=2`, but the domain function must still fail clearly if called directly with no inputs.
- Reject headers that do not match exactly.
- Allow empty SEGS payloads. They contribute no segments.
- If every input has no segments, return an empty SEGS payload with the shared header.
- Return an Impact-compatible tuple/list shape.

Error examples:

- No inputs:
  - `Batch SEGS requires one or more SEGS inputs.`
- Header mismatch:
  - `Batch SEGS requires all SEGS inputs to use the same image size; input 2 is 1024x768 but input 1 is 768x1024.`

Use height-first wording because SEGS headers are `(height, width)`.

## Batch SEGS Implementation Steps

### 1. Add Domain Function

Edit `simple_syrup/domain/segs.py`.

Status: completed.

Add a function similar to:

```python
def batch_segs(values: Iterable[object]) -> NativeSegs:
    """Return one SEGS payload containing all segments in input order."""
```

Implementation notes:

- Convert `values` to a tuple once so emptiness and indexing are deterministic.
- Use `coerce_segs` for each input.
- Track the first header as the expected header.
- Compare every later header to the first header.
- Extend a local `list[Segment]`.
- Return `(expected_header, tuple(segments))`.
- Do not call `to_impact_compatible_segs` in the domain function. Keep the domain return type native.
- Add `batch_segs` to imports where needed. Updating `__all__` is not necessary because this module does not currently define one.

### 2. Add V3 Node

Create `simple_syrup/nodes_v3/batch_segs.py`.

Status: completed.

Use the existing v3 import pattern from `tile_and_tag_segs.py`:

- Use `TYPE_CHECKING`.
- Import `comfy_api.latest` lazily through `import_module`.
- Define a `_ComfyNodeBase` type-checking shim.
- Use `_comfy_io.SEGS`.

Schema sketch:

```python
class BatchSEGSV3(_ComfyNodeBase):
    """Expose expandable SEGS batching through Comfy's v3 API."""

    @classmethod
    def define_schema(cls) -> Any:
        """Declare the Batch SEGS v3 schema."""

        autogrow_template = _comfy_io.Autogrow.TemplatePrefix(
            _comfy_io.SEGS.Input(
                "segs",
                tooltip="SEGS payload to append to the output batch.",
            ),
            prefix="segs",
            min=2,
            max=50,
        )
        return _comfy_io.Schema(
            node_id="SimpleSyrup.BatchSEGS",
            display_name="Batch SEGS",
            category="SimpleSyrup/Detection",
            description="Combines multiple SEGS inputs into one ordered SEGS payload.",
            search_aliases=["batch", "merge", "join", "combine", "segs"],
            inputs=[
                _comfy_io.Autogrow.Input(
                    "segs_inputs",
                    template=autogrow_template,
                    tooltip="Expandable SEGS inputs joined in socket order.",
                ),
            ],
            outputs=[
                _comfy_io.SEGS.Output(
                    "segs",
                    tooltip="Combined SEGS with all input segments in order.",
                ),
            ],
        )
```

Execute sketch:

```python
@classmethod
def execute(cls, segs_inputs: Any) -> tuple[object]:
    """Batch provided SEGS inputs in Autogrow order."""

    native = batch_segs(segs_inputs.values())
    return (to_impact_compatible_segs(native),)
```

Do not use `io.NodeOutput` unless existing SimpleSyrup v3 wrappers are migrated. Current local wrappers return normal tuples, so match local style.

### 3. Export V3 Node

Edit `simple_syrup/nodes_v3/__init__.py`.

Status: completed.

- Import `BatchSEGSV3` inside `get_nodes()`.
- Include `BatchSEGSV3` in both returned node lists:
  - prompt-control unavailable branch
  - prompt-control available branch
- Keep ordering sensible. Put it near `TileAndTagSEGSV3` because both are SEGS workflow utilities.

Do not edit `simple_syrup/nodes/__init__.py` for this v3-only node.

Do not edit root `__init__.py` unless tests prove the v3 entrypoint needs changes. It already delegates to `simple_syrup.nodes_v3.get_nodes()`.

## Batch SEGS Tests

### Domain Tests

Add tests to `tests/test_segs_domain.py`.

Status: completed.

Required cases:

1. Batches multiple SEGS payloads in order.
   - Input one has labels `1`, `2`, `3`.
   - Input two has labels `4`, `5`, `6`.
   - Output labels are `1`, `2`, `3`, `4`, `5`, `6`.

2. Accepts empty SEGS inputs.
   - Input one has labels `1`, `2`.
   - Input two is empty.
   - Input three has label `3`.
   - Output labels are `1`, `2`, `3`.

3. Returns empty output when all inputs are empty.
   - Same shared header.
   - Output segments tuple is empty.

4. Rejects no inputs.
   - `batch_segs(())` raises `ValueError`.

5. Rejects mismatched headers.
   - First input header `(8, 16)`.
   - Second input header `(16, 8)`.
   - Error message mentions the mismatched input index and both sizes.

Use existing `_segment(...)` helper in the test file where possible.

### V3 Node Tests

Create `tests/test_batch_segs_v3_node.py`.

Status: completed.

Required cases:

1. Schema test:
   - `node_id == "SimpleSyrup.BatchSEGS"`
   - `display_name == "Batch SEGS"`
   - category is `SimpleSyrup/Detection`
   - one input with id `segs_inputs`
   - output id is `segs`
   - output type is `SEGS`

2. Execute test:
   - Pass a dict like:
     ```python
     {"segs0": first, "segs1": second}
     ```
   - Assert returned labels preserve dict insertion order.
   - Assert return shape is Impact-compatible, meaning segments are a list.

3. Mismatched header test:
   - Call `BatchSEGSV3.execute(...)`.
   - Assert the domain error is surfaced.

### Registration Tests

Update `tests/test_registration.py`.

Status: completed.

In both v3 entrypoint tests, update expected node class names to include:

```text
BatchSEGSV3
```

Expected ordering should match `get_nodes()`.

No legacy mapping registration test is needed because this node is intentionally v3-only.

## WD14 SEGS Tagging Node

Status: completed after scope correction. The original document treated this as a follow-on, but the requested workflow included both **Batch SEGS** and **Tag SEGS w/ WD14**.

Implemented node name:

```text
Tag SEGS w/ WD14
```

Inputs:

- `image`: `IMAGE`
- `segs`: `SEGS`
- `clip`: `CLIP`
- `wd14_tagger`: `WD14_TAGGER`
- `universal_positive`: `STRING`
- `threshold`: `FLOAT`
- `character_threshold`: `FLOAT`
- `replace_underscore`: `BOOLEAN`
- `trailing_comma`: `BOOLEAN`
- `exclude_tags`: `STRING`

Outputs:

- `segs`: `SEGS`
- `positive`: `CONDITIONING_BATCH`

Implemented behavior:

- Validate the input image is a single image.
- Coerce incoming `SEGS`.
- Validate the `SEGS` header matches image dimensions.
- Crop each segment using `segment.crop_region`.
- Run WD14 over the crops in segment order.
- Prefix each tag prompt with the quality/universal positive prompt.
- Encode the resulting prompts using `ComfyConditioningEncoder.encode_batch`.
- Return the original SEGS unchanged plus the aligned `CONDITIONING_BATCH`.
- Reject mismatched tag count or conditioning count.

Implementation files:

- `simple_syrup/services/tag_segs_with_wd14_service.py`
- `simple_syrup/nodes/tag_segs_with_wd14.py`
- `simple_syrup/nodes_v3/tag_segs_with_wd14.py`
- `tests/test_tag_segs_with_wd14_service.py`
- `tests/test_tag_segs_with_wd14_node.py`
- `tests/test_tag_segs_with_wd14_v3_node.py`

## Batch Region Conditioning Node

Status: completed after the workflow requirement expanded to mixing autotagged SEGS conditioning with hand-authored regional conditioning. A v3 Autogrow node and a legacy two-input chainable node are both provided.

Implemented node name:

```text
Batch Region Conditioning
```

Inputs:

- Expandable v3 Autogrow inputs named `conditioning0`, `conditioning1`, etc.
- Legacy inputs named `first` and `second` for Comfy sessions that show legacy mapping nodes.
- Each input accepts either `CONDITIONING` or `CONDITIONING_BATCH`.
- Minimum visible/required inputs: `2`.
- Maximum inputs: `50`.

Output:

- `batch`: `CONDITIONING_BATCH`

Implemented behavior:

- Preserve socket order.
- If an input is `CONDITIONING_BATCH`, append all entries in that batch.
- If an input is normal `CONDITIONING`, append it as one entry.
- Return one flattened `CONDITIONING_BATCH`.

Example:

```text
auto_positive: [auto1, auto2]
hand_positive: hand1
extra_positive: [auto3]
        ↓
Batch Region Conditioning
        ↓
[auto1, auto2, hand1, auto3]
```

Implementation files:

- `simple_syrup/domain/conditioning_batch.py`
- `simple_syrup/nodes/batch_region_conditioning.py`
- `simple_syrup/nodes_v3/batch_region_conditioning.py`
- `tests/test_conditioning_batch.py`
- `tests/test_batch_region_conditioning_node.py`
- `tests/test_batch_region_conditioning_v3_node.py`

## Verification Commands

Run all commands from repository root:

```powershell
..\..\venv\Scripts\python.exe -m pytest -n auto -q tests\test_segs_domain.py tests\test_batch_segs_v3_node.py tests\test_registration.py
..\..\venv\Scripts\ruff.exe format .
..\..\venv\Scripts\ruff.exe check .
..\..\venv\Scripts\mypy.exe --strict simple_syrup tests
..\..\venv\Scripts\python.exe -m pytest -n auto -q
```

If a required tool is missing from `..\..\venv`, install or update development dependencies in that environment. Do not use global Python or a repository-local virtual environment.

## Definition of Done

- [x] `batch_segs` domain behavior exists and is tested.
- [x] **Batch SEGS** v3 node exists and uses `io.Autogrow`.
- [x] **Batch SEGS** is exported through `get_nodes()`.
- [x] Tests prove order-preserving flattening.
- [x] Tests prove mismatched headers fail clearly.
- [x] Tests prove v3 entrypoint includes the node.
- [x] Legacy node mappings are not changed for this v3-only node.
- [x] Required gates pass in the ComfyUI virtual environment.
- [x] **Tag SEGS w/ WD14** service, legacy node, and v3 node exist and are tested.
- [x] **Tag SEGS w/ WD14** is exported through legacy mappings and v3 entrypoint.
- [x] **Tag SEGS w/ WD14** emits `CONDITIONING_BATCH` from `CLIP` plus WD14 tags aligned to SEGS order.
- [x] **Batch Region Conditioning** v3 node accepts mixed `CONDITIONING` and `CONDITIONING_BATCH` inputs.
- [x] **Batch SEGS** and **Batch Region Conditioning** appear through legacy mappings for normal Comfy menu visibility.
