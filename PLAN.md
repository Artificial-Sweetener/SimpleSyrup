# Plan: Image-Associated Mask to SEGS

## Goal

Add a SimpleSyrup node that converts an existing ComfyUI mask into Impact-compatible SEGS while associating every SEG with cropped image data from the source image.

The node should feel like `Detect SEGS w/ Ultralytics`, but it must take an `IMAGE` and a `MASK` instead of an image and detector model. It should expose the same general region workflow controls where they make sense: size filtering, keep-only limiting, mask dilation, post-dilation, crop factor, sort order, and optional unioning into one SEG.

## Decisions From Maintainer Discussion

- This should be a first-class SimpleSyrup node, not just documentation for an Impact Pack workflow chain.
- The node should behave like a detector-style SEGS source because downstream users think of this as "pre-detected regions from a mask."
- The node must take both `image` and `mask`.
- The output SEGS must include full image association by storing each SEG's `cropped_image` from the input image.
- The node should support masks with multiple disconnected regions.
- The node should let users choose whether disconnected regions become separate SEGs or one combined SEG.
- The node should include controls similar to `Detect SEGS w/ Ultralytics` and in a similar order.
- The node should not expose detector confidence controls because the input is an existing mask, not detector predictions.
- Segment confidence should be fixed at `1.0`.
- The existing Impact Pack nodes can already do some of this with `MASK to SEGS` plus `Set Default Image for SEGS`, but that is not the desired UX for SimpleSyrup.

## Existing Code To Reuse

Use these SimpleSyrup modules as the main implementation references:

- `simple_syrup/nodes/detect_segs_with_ultralytics.py`
  - Existing detector-style node UX and control order.
  - Current output shape: `RETURN_TYPES = ("SEGS", "MASK")`.
  - Current list behavior for batched images: `OUTPUT_IS_LIST = (True, False)`.
  - Applies `limit_segs`, `sort_segs`, and then `build_combined_segs_result`.
- `simple_syrup/services/segs_detection_service.py`
  - Existing service pattern for constructing `Segment` objects from masks and image crops.
  - Uses `crop_region_for_bbox`, `crop_mask`, `crop_image`, and `dilate_mask`.
- `simple_syrup/services/segs_output_service.py`
  - `build_combined_segs_result()` already creates a one-SEG union with `cropped_image`.
  - `combined_mask_from_segs()` already produces the mask output from SEGS.
  - `coerce_cropped_mask()` validates crop-local SEG masks.
  - This module should become the shared owner for detector-style SEGS output finalization.
- `simple_syrup/masking/segs_mask_ops.py`
  - Reuse image validation, mask crop, image crop, resize, crop factor, and signed dilation behavior.
- `simple_syrup/domain/segs.py`
  - Reuse `Segment`, `BoundingBox`, `CropRegion`, `NativeSegs`, `SORT_ORDER_OPTIONS`, `limit_segs`, `sort_segs`, and `to_impact_compatible_segs`.

Use these Impact Pack modules only as behavioral references, not as imports:

- `E:\ComfyUI\custom_nodes\comfyui-impact-pack\modules\impact\segs_nodes.py`
  - `MaskToSEGS` splits or combines mask regions.
  - `DefaultImageForSEGS` attaches cropped source image data after mask conversion.
  - `SEGSMerge`, `SEGSOrderedFilter`, and `DilateMaskInSEGS` show existing user expectations.
- `E:\ComfyUI\custom_nodes\comfyui-impact-pack\modules\impact\core.py`
  - `mask_to_segs()` uses contour detection for disconnected regions.
  - `batch_mask_to_segs()` handles batched masks for video-style masks.

Do not import Impact Pack code. SimpleSyrup tests already enforce avoiding external pack imports.

## Node Name And Contract

Add a new Comfy v3 node:

- Node id: `SimpleSyrup.MaskToSEGS`
- Display name: `Mask to SEGS`
- Category: `SimpleSyrup/Detection`
- Search aliases: `mask`, `segs`, `region`, `detect`, `segmentation`
- Description: `Converts an existing mask into image-associated SEGS for detail and regional workflows.`

Outputs:

- `segs: SEGS`
- `mask: MASK`

Output behavior should match `Detect SEGS w/ Ultralytics`:

- `segs` is list-output compatible, one SEGS payload per input image/mask pair.
- `mask` is a standard ComfyUI batched mask tensor with shape `(B, H, W)`.
- The output mask is the union of the retained SEGS after filtering, sorting, and optional combining.

## Architecture Landing Shape

Do not add a third copy of the detector-style output pipeline.

The current code has two nodes that inline the same sequence after SEGS extraction:

1. `simple_syrup/nodes/detect_segs_with_ultralytics.py`
2. `simple_syrup/nodes/prompt_segs_with_sam.py`

Both nodes currently perform:

```python
segs = limit_segs(segs, keep_only, keep_by)
segs = sort_segs(segs, sort_order)
combined = build_combined_segs_result(single_image, segs, crop_factor)
output_segs = combined.segs if combine_segs else segs
segs_outputs.append(to_impact_compatible_segs(output_segs))
mask_outputs.append(combined.mask)
```

The new mask node must not duplicate this sequence inline. Instead, implement a shared output-finalization helper in `simple_syrup/services/segs_output_service.py` and refactor the existing detector nodes to use it.

Recommended shape:

- `MaskToSEGSService`
  - Owns only one-image, one-mask extraction into separate native SEGS.
  - Does not own `keep_only`, `sort_order`, `combine_segs`, Impact-compatible conversion, or output mask construction.
- `segs_output_service`
  - Owns the common detector-style post-processing pipeline:
    1. limit SEGS
    2. sort SEGS
    3. build the combined result
    4. choose separate or combined output SEGS
    5. convert output SEGS to Impact-compatible shape
    6. return the paired mask output
- Node classes
  - Own Comfy-facing schema, batch iteration, and wiring only.
  - Delegate source-specific extraction to a source service.
  - Delegate shared final output shaping to `segs_output_service`.

This keeps ownership strict:

- Mask-derived region extraction belongs to the mask-to-SEGS service.
- Detector model inference belongs to detector services.
- Prompt/SAM detection belongs to the SAM prompt service.
- Sorting, limiting, combining, and output mask generation belong to the shared SEGS output service.
- Domain helpers remain pure SEGS policies and value conversions.

## Input Schema

Use this input order:

1. `image: IMAGE`
2. `mask: MASK`
3. `mask_threshold: FLOAT`
4. `size_threshold: INT`
5. `keep_only: INT`
6. `mask_dilation: INT`
7. `post_dilation: INT`
8. `crop_factor: FLOAT`
9. `sort_order: SORT_ORDER_OPTIONS`
10. `combine_segs: BOOLEAN`
11. `label: STRING`

Detailed input behavior:

- `image`
  - Source image used to associate cropped image data with every SEG.
  - Must be a ComfyUI `IMAGE` tensor shaped `(B, H, W, C)`.
- `mask`
  - Source mask to convert into SEGS.
  - Must be a ComfyUI `MASK` tensor shaped `(B, H, W)` or a single mask compatible with the image batch.
  - The implementation may support a single mask for a single image first. If supporting image batches, batch count must either match the image batch or be exactly `1` for reuse across all images.
- `mask_threshold`
  - Default: `0.5`
  - Min: `0.0`
  - Max: `1.0`
  - Step: `0.01`
  - Converts soft masks into active pixels before region extraction.
  - Active pixels are `mask >= mask_threshold`.
- `size_threshold`
  - Default: `10`
  - Min: `1`
  - Max: `8192`
  - Drops extracted regions whose bounding box is smaller than this many pixels wide or tall.
  - Match the wording and role of `Detect SEGS w/ Ultralytics`'s `size_threshold`.
- `keep_only`
  - Default: `0`
  - Min: `0`
  - Max: `4096`
  - Keeps only the largest N regions after size filtering. `0` keeps all regions.
  - Do not expose a confidence ranking option. There is no detector confidence.
- `mask_dilation`
  - Default: `0`
  - Min: `-512`
  - Max: `512`
  - Signed dilation applied to the full input mask before region extraction.
  - Positive grows mask regions; negative erodes them.
- `post_dilation`
  - Default: `0`
  - Min: `-512`
  - Max: `512`
  - Signed dilation applied to each final crop-local SEG mask after the crop is chosen.
  - This mirrors the Ultralytics node's final SEG mask cleanup behavior.
- `crop_factor`
  - Default: `3.0`
  - Min: `0.0`
  - Max: `100.0`
  - Step: `0.1`
  - `0.0` means use the full image as the crop, matching SimpleSyrup's existing detector convention.
  - Values greater than or equal to `1.0` expand the SEG crop around the extracted region's bounding box.
  - Values between `0.0` and `1.0` must fail with an actionable `ValueError`.
- `sort_order`
  - Use `SORT_ORDER_OPTIONS` from `simple_syrup.domain.segs`.
  - Default: `largest to smallest`.
  - Sort extracted regions before output and before any combined mask result is built.
- `combine_segs`
  - Default: `False`
  - `False` returns one SEG per disconnected mask region.
  - `True` returns one unioned SEG representing all retained mask pixels.
- `label`
  - Default: `mask`
  - Label applied to extracted SEGs.
  - If `combine_segs` is true, the combined output label should be `combined` unless there is a strong reason to preserve the user label. Match `build_combined_segs_result()` unless intentionally extending it.

## Behavior Details

### Region Extraction

Add a new service, probably `simple_syrup/services/mask_to_segs_service.py`, with a class such as `MaskToSEGSService`.

The service should own:

- Image and mask validation.
- Mask thresholding.
- Signed full-mask dilation.
- Disconnected region extraction.
- Per-region bounding box calculation.
- Size filtering.
- Crop region calculation.
- Crop-local mask creation.
- Optional post-dilation on each crop-local mask.
- Cropped image association.
- Native immutable SEGS output.

The service must not own:

- `keep_only`
- final `sort_order`
- `combine_segs`
- Impact-compatible conversion
- final output mask construction

Those responsibilities belong to the shared SEGS output finalization helper.

The node should own only Comfy-facing schema, input ordering, batch iteration, and wiring.

### Connected Components

Implement disconnected region extraction inside SimpleSyrup. Do not import Impact Pack.

Prefer a torch-based or standard-library implementation over adding a new required runtime dependency. A simple deterministic flood-fill or connected-components routine is acceptable because masks are 2D binary tensors and the node is not model-bound.

Connectivity decision:

- Use 8-connected components unless tests or existing SimpleSyrup behavior strongly point to 4-connected components.
- Document this in the service docstring and tests.
- 8-connected behavior usually matches user expectations for painted masks where diagonal contact should remain one region.

Extraction algorithm outline:

1. Convert mask to a CPU `torch.bool` active-pixel tensor after threshold and dilation.
2. If no pixels are active, return empty native SEGS. The shared output finalization helper is responsible for turning that into a zero mask output.
3. Find connected components.
4. For each component, compute bbox from active pixel coordinates.
5. Drop components where bbox width or height is less than `size_threshold`.
6. Create one `Segment` per kept component.

The service always returns separate native SEGS. `combine_segs` is handled later by the shared output finalization helper so all detector-style SEGS nodes use the same combine behavior.

### Mask Values

Use thresholded masks to decide region membership, but preserve useful soft-mask values when forming crop-local masks where possible.

Recommended behavior:

- Use the thresholded, dilated mask for topology and bbox extraction.
- Use the original normalized mask after `mask_dilation` as the crop-local mask values.
- Zero out pixels outside the active component for each separate SEG.
- Clamp all final masks to `0.0..1.0`.

This lets soft masks retain feathered values inside each SEG while still giving deterministic region extraction.

### Cropped Image Association

Every returned `Segment` must set:

- `cropped_image = crop_image(single_image, crop_region).detach().clone()`
- `cropped_mask = crop-local mask tensor`
- `confidence = 1.0`
- `crop_region = CropRegion(...)`
- `bbox = BoundingBox(...)`
- `label = label`
- `control_net_wrapper = None`

This is the key difference from Impact Pack's `MASK to SEGS`, which creates SEGS without cropped image data and relies on a separate `Set Default Image for SEGS` node.

### Batch Handling

Match existing SimpleSyrup detector behavior as closely as possible:

- Validate the image batch with `validate_image_batch()`.
- Iterate image items with `iter_single_images()`.
- Produce one SEGS payload per image.
- Concatenate mask outputs into one `(B, H, W)` tensor.

Mask batch rules:

- If mask batch size equals image batch size, pair by index.
- If mask batch size is `1` and image batch size is greater than `1`, reuse the mask for every image.
- Otherwise raise `ValueError` explaining the mismatch.
- Mask height and width must match the image height and width. Do not silently resize masks for this node unless the maintainer explicitly approves it later.

## Sorting And Keeping

Do not expose `keep_by` unless there is a future product decision to add multiple non-confidence policies.

For `keep_only`, retain the largest regions by crop area before final sorting. Reuse `limit_segs(segs, keep_only, "largest size")` inside the shared output finalization helper for this node. Do not add a new ranking abstraction unless tests show the current domain helper cannot express the behavior clearly.

Final output order must come from `sort_segs(segs, sort_order)`.

## Shared Output Finalization

Add a small result type and helper to `simple_syrup/services/segs_output_service.py`.

Suggested result type:

```python
@dataclass(frozen=True)
class FinalizedSegsOutput:
    """Return Impact-compatible SEGS and its paired output mask."""

    segs: object
    mask: torch.Tensor
```

Suggested helper:

```python
def finalize_detector_segs_output(
    image: object,
    segs: NativeSegs,
    keep_only: int,
    keep_by: str,
    crop_factor: float,
    sort_order: str,
    combine_segs: bool,
) -> FinalizedSegsOutput:
    """Apply shared detector-style SEGS output policy."""
```

Behavior:

1. Apply `limit_segs(segs, keep_only, keep_by)`.
2. Apply `sort_segs(segs, sort_order)`.
3. Build `combined = build_combined_segs_result(image, segs, crop_factor)`.
4. Use `combined.segs` when `combine_segs` is true; otherwise use the sorted separate SEGS.
5. Convert chosen SEGS with `to_impact_compatible_segs`.
6. Return the converted SEGS and `combined.mask`.

For the new mask node, call this helper with `keep_by="largest size"` internally.

Refactor these existing nodes to use the helper as part of this change:

- `simple_syrup/nodes/detect_segs_with_ultralytics.py`
- `simple_syrup/nodes/prompt_segs_with_sam.py`

Add characterization tests before refactoring or preserve existing tests that already prove:

- final sorting happens before combined output construction
- keep-only limiting happens before final sorting
- `combine_segs` chooses the combined SEGS while retaining the same output mask
- returned SEGS are Impact-compatible

The helper should be narrow. Do not move source-specific detection behavior into it.

## V3 Registration

Comfy v3 is the only supported export path.

Preferred implementation:

- Add a direct v3 node class under `simple_syrup/nodes_v3/mask_to_segs.py`.
- Register it in `simple_syrup/nodes_v3/__init__.py::get_nodes()`.
- Do not add `NODE_CLASS_MAPPINGS` or legacy export mappings.

If reusing the legacy adapter pattern would materially reduce risk, it is acceptable to add a legacy-style internal node class under `simple_syrup/nodes/` and wrap it with `LegacyNodeV3Adapter`, but the public export must still be v3-only.

The implementation should look native to the current codebase and should not add compatibility shims.

## Tooltips

Every visible input and output must have concise user-facing tooltip text.

Required tooltip intent:

- `image`: source image used for SEG crops.
- `mask`: mask whose active regions become SEGS.
- `mask_threshold`: threshold used to decide active mask pixels.
- `size_threshold`: smallest region width or height to keep, in pixels.
- `keep_only`: maximum number of largest regions to keep; `0` keeps all.
- `mask_dilation`: grow or shrink the source mask before regions are found.
- `post_dilation`: grow or shrink each final SEG mask after cropping.
- `crop_factor`: context around each region; `0` uses the full image.
- `sort_order`: output ordering for separate SEGS.
- `combine_segs`: return one unioned SEG instead of separate regions.
- `label`: label stored on extracted SEGs.
- `segs` output: image-associated SEGS from the mask.
- `mask` output: union of retained SEGS as a ComfyUI mask.

## Tests To Add

Add focused tests before or alongside implementation.

### Shared Output Finalization Tests

Add tests for the new helper in `tests/test_segs_output_service.py` or extend the existing SEGS output service coverage currently housed in `tests/test_ultralytics_detection_service.py`.

Cover:

- Applies `limit_segs` before `sort_segs`.
- Builds the combined result from the limited and sorted SEGS.
- Returns separate SEGS when `combine_segs` is false.
- Returns one combined SEG when `combine_segs` is true.
- Always returns the mask from `build_combined_segs_result`.
- Converts output SEGS to Impact-compatible tuple/list shape.
- Supports `keep_by="largest size"` for mask-derived SEGS.

After this helper is tested, refactor `Detect SEGS w/ Ultralytics` and `Prompt SEGS w/ SAM` to use it without changing their public behavior.

### Service Tests

Create `tests/test_mask_to_segs_service.py`.

Cover:

- Single rectangular mask creates one SEG.
- `cropped_image` matches the source image crop.
- `cropped_mask` matches the mask crop.
- Two disconnected regions create two separate SEGs.
- Service always returns separate SEGS; combining is covered by shared output finalization tests and node tests.
- Empty mask returns empty native SEGS from the service and a zero mask output through the node/output helper.
- `mask_threshold` controls active pixels.
- `mask_dilation` grows a source mask before region extraction.
- Negative `mask_dilation` erodes a source mask.
- `post_dilation` changes only crop-local final masks.
- `crop_factor` expands crop regions.
- `crop_factor = 0.0` uses the full image.
- `0.0 < crop_factor < 1.0` raises `ValueError`.
- `size_threshold` drops small components.
- Soft mask values are clamped and retained inside component masks.
- Batch mask and image mismatch raises an actionable error if batch handling is in the service.

### Node Contract Tests

Create `tests/test_mask_to_segs_node.py` or a v3-specific equivalent.

Cover:

- Node id is `SimpleSyrup.MaskToSEGS`.
- Display name is `Mask to SEGS`.
- Category is `SimpleSyrup/Detection`.
- Outputs are `SEGS` and `MASK`.
- The input list order exactly matches the plan.
- Every input has a tooltip.
- Every output has a tooltip.
- No confidence input exists.
- No detector model input exists.
- No `keep_by` widget exists unless the implementation deliberately adds more non-confidence policies and updates this plan.
- Execution returns Impact-compatible SEGS and a `(B, H, W)` mask.
- `combine_segs = false` returns separate SEGs.
- `combine_segs = true` returns one combined SEG.
- `keep_only` keeps the largest mask-derived regions.
- `sort_order` orders separate regions using existing domain policies.
- Batched images return list-output SEGS and batched masks.

### Registration Tests

Update existing registration tests:

- `SimpleSyrup.MaskToSEGS` appears in `get_nodes()`.
- Adding the node does not remove or rename existing nodes.
- Root `comfy_entrypoint` still exposes v3 nodes only.

### Tooltip Coverage Tests

Update `tests/test_node_tooltips.py` so the new node passes the repository tooltip requirements.

## Implementation Steps

- [x] Add or confirm characterization tests for the existing Ultralytics and SAM detector-style output behavior.
  - Landing note: Existing node tests already covered separate/combined outputs, keep-only limiting, final sorting, combined-builder input order, and batch handling for both detector nodes.
- [x] Add shared output finalization tests.
  - Landing note: `tests/test_segs_output_service.py` now covers limit-before-sort, combined output selection, returned mask preservation, Impact-compatible conversion, and `keep_by="largest size"` for mask-derived SEGS.
- [x] Add the shared output finalization helper in `segs_output_service.py`.
  - Landing note: `FinalizedSegsOutput` and `finalize_detector_segs_output()` now own detector-style limit/sort/combine/conversion/mask finalization.
- [x] Refactor `Detect SEGS w/ Ultralytics` and `Prompt SEGS w/ SAM` to use the helper without behavior changes.
  - Landing note: Both nodes still own source-specific extraction and batch wiring, but delegate shared output shaping to `finalize_detector_segs_output()` with their injectable combined builders.
- [x] Add the connected-component helper and mask-to-SEGS service.
  - Landing note: `mask_components.py` implements deterministic 8-connected component extraction, and `MaskToSEGSService` now converts one image plus one mask into separate native SEGS with cropped image association.
- [x] Add direct v3 node schema and execution wrapper.
  - Landing note: `MaskToSEGSV3` defines the Comfy v3 schema directly, owns image/mask batch pairing, and delegates extraction plus shared output finalization.
- [x] Register the node in `simple_syrup/nodes_v3/__init__.py`.
  - Landing note: `SimpleSyrup.MaskToSEGS` is included in the base v3 node list.
- [x] Add or update registration and tooltip tests.
  - Landing note: Registration expectations include `SimpleSyrup.MaskToSEGS`; the schema-driven tooltip coverage includes every new input and output.
- [x] Run focused tests for shared output finalization, existing detector nodes, the new service, the new node, registration, and tooltips.
  - Landing note: Focused plan checks pass: `51 passed`.
- [x] Run full Python gates.
  - Landing note: `ruff format .`, `ruff check .`, `mypy --strict simple_syrup tests`, and full `pytest -n auto -q` pass. Full test result: `969 passed`.
- [x] Do not touch frontend code unless the Comfy v3 UI requires it.
  - Landing note: No frontend source or generated browser artifact was changed.

## Verification Commands

Run all commands from repository root with the ComfyUI virtual environment two directories above this repo.

Focused checks during development:

```powershell
..\..\venv\Scripts\python.exe -m pytest -n auto -q tests\test_detect_segs_with_ultralytics_node.py tests\test_prompt_segs_with_sam_node.py
..\..\venv\Scripts\python.exe -m pytest -n auto -q tests\test_segs_output_service.py
..\..\venv\Scripts\python.exe -m pytest -n auto -q tests\test_mask_to_segs_service.py tests\test_mask_to_segs_node.py
..\..\venv\Scripts\python.exe -m pytest -n auto -q tests\test_registration.py tests\test_node_tooltips.py
```

Required final gates:

```powershell
..\..\venv\Scripts\ruff.exe format .
..\..\venv\Scripts\ruff.exe check .
..\..\venv\Scripts\mypy.exe --strict simple_syrup tests
..\..\venv\Scripts\python.exe -m pytest -n auto -q
```

If frontend code is touched, also run:

```powershell
npm ci
npm run lint:web
npm run typecheck:web
npm run test:web
npm run build:web
```

## Acceptance Criteria

- A user can plug in an image and a mask and get ready-to-use image-associated SEGS.
- A mask with two disconnected regions can produce either two SEGs or one combined SEG.
- The node has detector-style controls that feel aligned with `Detect SEGS w/ Ultralytics`.
- The node exposes no detector confidence controls.
- Each SEG has `cropped_image`, `cropped_mask`, `crop_region`, `bbox`, `label`, and `confidence = 1.0`.
- The returned SEGS are Impact-compatible.
- The returned mask is the union of retained output SEGS.
- The implementation does not import Impact Pack.
- The implementation is covered by behavior tests, node contract tests, registration tests, and tooltip tests.
- Full Python verification gates pass.
