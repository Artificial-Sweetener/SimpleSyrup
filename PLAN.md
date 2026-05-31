# Plan: Tag SEGS w/ External LLM

## Goal

Add a new Comfy v3-only node named `Tag SEGS w/ External LLM`.

The node will take an image, existing Impact-compatible SEGS, and a CLIP object. It will show each SEG crop to the configured external vision LLM, turn each LLM response into a positive regional prompt, CLIP-encode those prompts in the same order as the SEGS, and return the unchanged SEGS plus an aligned `CONDITIONING_BATCH` output named `positive`.

The intended workflow is the same regional-conditioning role currently served by `Tag SEGS w/ WD14` and `Tile & Tag SEGS`, but with an external vision model instead of WD14. The node should be useful when a vision LLM can produce better region descriptions than WD14 tags.

## Decisions From Planning

- Export only through the Comfy v3 node path.
- Do not add a legacy node class.
- Do not add `NODE_CLASS_MAPPINGS` or `NODE_DISPLAY_NAME_MAPPINGS`.
- Do not add an internal compatibility shim or dual implementation path.
- Make exactly one external LLM call per SEG.
- Preserve SEGS order exactly.
- Return the original SEGS unchanged, converted only to the existing Impact-compatible output shape.
- Output one `CONDITIONING_BATCH` entry per SEG.
- Fail closed when alignment is uncertain.
- Provide a single widget for how SEG crops are presented to the vision model.
- Support LLM response cleanup controls similar to the WD14 tagging nodes.
- Treat the LLM response as prompt text after lightweight tag formatting. Do not add clever semantic parsing in the first implementation.
- Do not provide opinionated default prompt text for `system_prompt` or `user_prompt`; leave both defaults empty.

## Implementation Status

- [x] Review plan and current code patterns.
  - Landing note: existing `Tag SEGS w/ WD14` owns SEGS validation/alignment, `External LLM Prompt` owns provider settings/key flow, and v3 registration is centralized in `simple_syrup/nodes_v3/__init__.py`.
- [x] Add SEG crop image encoder and tests.
  - Landing note: `ExternalLLMSegsImageEncoder` now emits PNG data URLs for `transparent mask`, `black mask`, and `full crop`; focused encoder tests pass.
- [x] Add external LLM image-data-url execution path.
  - Landing note: `ExternalLLMPromptService.generate_with_image_data_url()` reuses the existing model/settings/key/client flow while accepting the SEG encoder's prebuilt PNG data URL.
- [x] Add service formatting and orchestration tests.
  - Landing note: service tests cover per-SEG LLM call order, prompt formatting, exclusions, progress, logging metadata, validation failures, and conditioning count alignment.
- [x] Add `TagSEGSWithExternalLLMService`.
  - Landing note: the service owns SEGS validation, SEG image encoding, pre-encoded LLM calls, response formatting, prompt prefixing, CLIP conditioning encoding, and completion logging.
- [x] Add v3-only node schema and execution forwarding.
  - Landing note: `TagSEGSWithExternalLLMV3` defines the schema directly and delegates to `TagSEGSWithExternalLLMService`; no file was added under `simple_syrup/nodes/`. `system_prompt` and `user_prompt` default to empty strings.
- [x] Register v3 node and update registration tests.
  - Landing note: `SimpleSyrup.TagSEGSWithExternalLLM` is now returned by `get_nodes()` with the maintained base node list.
- [x] Add tooltips and update tooltip coverage.
  - Landing note: all new v3 inputs/outputs use dedicated tooltip constants, and focused registration/tooltip tests pass.
- [x] Update this plan with final landing notes.
  - Landing note: implementation completed as a v3-only vertical slice with no new `simple_syrup/nodes/` node class and no legacy mapping export changes.
- [x] Run required verification gates.
  - Landing note: `ruff format .`, `ruff check .`, `mypy --strict simple_syrup tests`, and `pytest -n auto -q` pass in `..\..\venv`.

## New Node Contract

Node id:

```text
SimpleSyrup.TagSEGSWithExternalLLM
```

Display name:

```text
Tag SEGS w/ External LLM
```

Category:

```text
SimpleSyrup/Detailing
```

Inputs:

```text
image: IMAGE
segs: SEGS
clip: CLIP
model: external LLM model dropdown
system_prompt: STRING
user_prompt: STRING
universal_positive: STRING
seg_image_mode: COMBO
replace_underscore: BOOLEAN
trailing_comma: BOOLEAN
exclude_tags: STRING
max_tokens: INT
reasoning_effort: COMBO
```

Outputs:

```text
segs: SEGS
positive: CONDITIONING_BATCH
```

Suggested default input values:

```text
system_prompt:

user_prompt:

universal_positive:

seg_image_mode:
transparent mask

replace_underscore:
true

trailing_comma:
false

exclude_tags:

max_tokens:
1024

reasoning_effort:
default
```

Search aliases:

```text
llm
vision
tag
segs
detail
regional
prompt
```

## SEG Image Modes

Add one combo input named `seg_image_mode` with these exact options:

```text
transparent mask
black mask
full crop
```

### `transparent mask`

Crop the original image to `segment.crop_region`.

Crop the SEG mask to the same rectangle. Resize or normalize the mask only if needed to match the crop dimensions.

Encode a PNG data URL with RGBA channels:

- RGB comes from the crop.
- Alpha comes from the cropped SEG mask.
- Pixels outside the mask are hidden by alpha.

Use this as the default because it most directly communicates "only this SEG matters" when the provider supports alpha.

### `black mask`

Crop the original image to `segment.crop_region`.

Crop the SEG mask to the same rectangle. Resize or normalize the mask only if needed to match the crop dimensions.

Encode an RGB PNG data URL:

- Pixels inside the mask keep the original crop color.
- Pixels outside the mask are set to black.

This is the compatibility mode for providers that ignore PNG alpha or flatten transparent pixels unpredictably.

### `full crop`

Crop the original image to `segment.crop_region`.

Encode the full RGB crop unchanged.

Ignore the SEG mask for the image sent to the LLM. This gives the model surrounding context when context improves region tagging.

## Formatting Rules For LLM Responses

Add formatting controls:

```text
replace_underscore: BOOLEAN = true
trailing_comma: BOOLEAN = false
exclude_tags: STRING = ""
```

For each LLM response:

1. Strip leading and trailing whitespace.
2. Reject the response if it is empty after stripping.
3. Split the response by commas into tag-like chunks.
4. Trim whitespace around each chunk.
5. Drop empty chunks.
6. If `replace_underscore` is enabled, replace `_` with spaces inside each chunk.
7. Parse `exclude_tags` as a comma-separated list.
8. Normalize excluded tags using the same trim, empty-drop, and optional underscore replacement policy.
9. Remove exact tag matches after normalization.
10. Rejoin remaining tags with `, `.
11. Reject the prompt if no tags remain after exclusion.
12. If `trailing_comma` is enabled and the prompt does not already end in a comma, append `,`.
13. Prefix with `universal_positive` using `domain.prompt_composition.prefix_prompt`.

Do not fuzzy-match excluded tags.

Do not parse JSON, markdown, prose sections, or model-specific response schemas in the first implementation. Prompt discipline should keep responses comma-like. If a model returns prose, the comma splitting still gives deterministic behavior.

## Architecture

Follow the existing layer boundaries in `AGENTS.md`.

### Domain Layer

Use existing domain objects where possible:

- `simple_syrup/domain/segs.py`
- `simple_syrup/domain/conditioning_batch.py`
- `simple_syrup/domain/prompt_composition.py`
- `simple_syrup/domain/external_llm.py`

Add a small domain value object only if it removes duplication or clarifies validation. A likely useful object is a frozen dataclass for formatting controls:

```python
@dataclass(frozen=True)
class LLMTagFormattingControls:
    replace_underscore: bool
    trailing_comma: bool
    exclude_tags: str
```

Place it where ownership is clearest after implementation. If the formatting is used only by this service, keep it service-local. If it becomes shared with future LLM tagging code, place it in a domain module.

### Runtime / Adapter Layer

Add a crop image encoder near:

```text
simple_syrup/runtime/external_llm_images.py
```

Recommended additions:

```python
SEG_IMAGE_MODES = ("transparent mask", "black mask", "full crop")

class ExternalLLMSegsImageEncoder:
    """Encode SEG crops for OpenAI-compatible vision payloads."""

    def encode_segment_as_data_url(
        self,
        image: torch.Tensor,
        segment: Segment,
        mode: str,
    ) -> str:
        """Return one SEG crop as a PNG data URL."""
```

This adapter owns tensor-to-PIL conversion and PNG/base64 data URL encoding.

The existing `ExternalLLMImageEncoder.encode_first_image_as_data_url()` sends the first whole image. Do not overload that method with SEG-specific behavior. Keep the SEG crop behavior separate so the existing `External LLM Prompt` node remains simple and unchanged.

Use existing masking helpers where possible:

- `validate_single_image`
- `crop_image`
- `crop_mask`
- `resize_mask`

If `segment.cropped_mask` is already crop-local, do not assume it is full-image. The service or encoder must normalize the mask robustly:

- If the mask shape matches the crop height and width, use it directly.
- If the mask shape matches the full source image height and width, crop it by `segment.crop_region`.
- If the mask has a batch dimension, normalize to one HW mask.
- If the mask shape is compatible but not exact, resize with bilinear interpolation and clamp to `0.0..1.0`.
- If the mask cannot be interpreted as HW or BHW, raise a clear `ValueError`.

Preserve security boundaries:

- No filesystem writes are needed.
- No workflow-provided code execution.
- No shell invocation.
- No network logic inside the image encoder.

### Application / Service Layer

Add a service module:

```text
simple_syrup/services/tag_segs_with_external_llm_service.py
```

Recommended public result:

```python
@dataclass(frozen=True)
class TagSEGSWithExternalLLMResult:
    """Return unchanged SEGS and aligned external-LLM conditioning."""

    segs: ImpactSegs
    positive: ConditioningBatch
```

Recommended service:

```python
class TagSEGSWithExternalLLMService:
    """Caption provided SEGS crops with an external LLM and encode conditioning."""
```

The service should own:

- Input validation.
- External LLM model resolution.
- External LLM settings and credential checks.
- One provider request per SEG.
- Progress reporting.
- Response formatting.
- Prompt prefixing.
- CLIP conditioning encoding.
- Alignment checks.
- Structured logging.

Keep external boundaries injectable for tests:

```python
class VisionLLMBoundary(Protocol):
    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        reasoning_effort: str,
        image: object | None = None,
    ) -> str:
        """Return one assistant response."""


class SegmentImageEncodingBoundary(Protocol):
    def encode_segment_as_data_url(
        self,
        image: torch.Tensor,
        segment: Segment,
        mode: str,
    ) -> str:
        """Return one SEG crop image data URL."""


class ConditioningEncodingBoundary(Protocol):
    def encode_batch(self, clip: Any, chunks: tuple[str, ...]) -> ConditioningBatch:
        """Return conditioning entries in prompt order."""
```

The existing `ExternalLLMPromptService.generate()` expects an `image` object and then asks its own image encoder to encode it. For this node, prefer adding a provider-call method that accepts a prebuilt image data URL, or extract shared provider execution into a helper. Do not pass already encoded data URLs through an API that claims to accept Comfy `IMAGE` tensors.

Recommended clean shape:

- Keep `ExternalLLMPromptService.generate()` behavior unchanged.
- Add a method to the same service or a new small collaborator that accepts `image_data_url`.
- Reuse the same settings repository, key store, model choice, and provider client logic.
- Avoid duplicating endpoint/API-key validation across services if it can be extracted without creating an unclear abstraction.

The service execution should look like:

1. Validate `image` as a single BHWC IMAGE tensor.
2. Coerce `segs` with `coerce_segs`.
3. Validate SEGS header matches the image height and width.
4. Validate every `segment.crop_region` fits inside the image.
5. Reject empty SEGS.
6. Resolve the selected LLM model.
7. Create progress with `len(segments) + 2`.
8. For each segment in input order:
   - Encode one SEG crop according to `seg_image_mode`.
   - Call the external LLM once with that crop.
   - Format the LLM response.
   - Prefix with `universal_positive`.
   - Append the prompt to an ordered tuple/list.
   - Update progress.
9. Encode the prompt tuple using `ComfyConditioningEncoder.encode_batch`.
10. Verify `len(positive.entries) == len(segments)`.
11. Log completion with operation name, segment count, selected model, image mode, and formatting-control presence.
12. Return `to_impact_compatible_segs(native_segs)` and `positive`.

Suggested operation constant:

```python
OPERATION = "Tag SEGS w/ External LLM"
```

### Comfy v3 Node Layer

Add only:

```text
simple_syrup/nodes_v3/tag_segs_with_external_llm.py
```

Do not add:

```text
simple_syrup/nodes/tag_segs_with_external_llm.py
```

The v3 node should:

- Define the schema directly.
- Use `comfy_api.latest`.
- Use `Custom("CONDITIONING_BATCH")` for the positive output.
- Use model choices from the external LLM service, like the current `External LLM Prompt` node does.
- Delegate non-trivial behavior to `TagSEGSWithExternalLLMService`.
- Contain only input declaration and execution forwarding.

Register the node in:

```text
simple_syrup/nodes_v3/__init__.py
```

The root package export must continue to expose only `comfy_entrypoint`.

## Tooltips

Update:

```text
simple_syrup/nodes/tooltips.py
```

Every visible input and output needs a concise user-facing tooltip.

Suggested tooltip intent:

- `image`: The source image that the SEGS were detected from.
- `segs`: The regions to describe and align with conditioning.
- `clip`: The CLIP/text encoder used to encode generated regional prompts.
- `model`: The configured external vision model used for each SEG crop.
- `system_prompt`: Instructions that control how the model writes tags.
- `user_prompt`: The per-region request sent with each SEG crop.
- `universal_positive`: Prompt text added before every generated region prompt.
- `seg_image_mode`: How pixels outside the SEG mask are shown to the vision model.
- `replace_underscore`: Converts booru-style underscores into spaces before encoding.
- `trailing_comma`: Adds a final comma to each generated prompt when enabled.
- `exclude_tags`: Comma-separated exact tags to remove from LLM responses.
- `max_tokens`: Maximum response length for each SEG request.
- `reasoning_effort`: Provider-specific reasoning control, when supported.
- `segs` output: The original SEGS, kept in the same order as the conditioning batch.
- `positive` output: One CLIP-encoded positive conditioning entry per SEG.

Keep wording concise. Do not document removed choices or alternatives the node does not expose.

## Tests

Add or update tests alongside implementation.

### Image Encoder Tests

Add:

```text
tests/test_external_llm_segs_images.py
```

Cover:

- `transparent mask` returns a PNG data URL with alpha.
- Alpha is low/zero outside the SEG mask and high/one inside it.
- `black mask` returns RGB where outside-mask pixels are black.
- `full crop` preserves the crop rectangle without masking.
- Crop dimensions match `segment.crop_region`.
- Crop-local masks are accepted.
- Full-image masks are accepted and cropped.
- Invalid mask rank fails with an actionable `ValueError`.
- Unknown `seg_image_mode` fails with an actionable `ValueError`.

Use small deterministic tensors, such as 4x4 or 6x6 images, so pixel assertions are simple.

### Service Tests

Add:

```text
tests/test_tag_segs_with_external_llm_service.py
```

Cover:

- Existing SEGS, LLM responses, formatted prompts, and conditioning stay aligned by index.
- The service makes one LLM call per SEG.
- The service sends the selected `seg_image_mode` to the encoder.
- `universal_positive` is prefixed to every generated prompt.
- `replace_underscore=True` turns `blue_hair` into `blue hair`.
- `replace_underscore=False` preserves `blue_hair`.
- `exclude_tags` removes exact normalized tags.
- `trailing_comma=True` adds a comma after formatting.
- Empty SEGS raise `ValueError`.
- Mismatched SEGS header and image dimensions raise `ValueError`.
- Crop regions outside the image raise `ValueError`.
- Empty LLM responses raise `ValueError`.
- Responses that become empty after exclusions raise `ValueError`.
- Conditioning encoder returning the wrong entry count raises `ValueError`.
- Completion logging includes operation, segment count, model, image mode, and universal-positive presence.

Use fake boundaries for LLM calls, image encoding, conditioning encoding, and progress. Do not call a real provider in tests.

### V3 Node Tests

Add:

```text
tests/test_tag_segs_with_external_llm_v3_node.py
```

Cover:

- Schema node id is `SimpleSyrup.TagSEGSWithExternalLLM`.
- Display name is `Tag SEGS w/ External LLM`.
- Category is `SimpleSyrup/Detailing`.
- Inputs are exposed in the intended order.
- `seg_image_mode` choices are exactly `transparent mask`, `black mask`, `full crop`.
- `transparent mask` is the default.
- Formatting controls have intended defaults.
- Outputs are `segs` and `positive`.
- Output types are `SEGS` and `CONDITIONING_BATCH`.
- `execute()` forwards all inputs to the service.

### Registration Tests

Update existing registration/v3 export tests. The new node must appear in the v3 entrypoint output from:

```text
simple_syrup/nodes_v3/__init__.py::get_nodes()
```

Also update tooltip coverage tests so the new node does not create coverage gaps.

## Validation And Errors

Use explicit, actionable errors.

Required validation failures:

- `image` is not a torch IMAGE tensor.
- `image` batch size is not 1.
- `segs` is not a valid SEGS payload.
- SEGS header dimensions do not match the image.
- SEGS contains no segments.
- A SEG crop region is outside the image.
- `seg_image_mode` is unknown.
- The external LLM provider is not configured.
- The external LLM API key is missing.
- The external LLM response is empty.
- Formatting removes every generated tag.
- The conditioning encoder returns a count different from the SEG count.

Do not silently continue after invalid inputs or provider failures.

## Logging And Progress

Use `simple_syrup.shared.logging.get_logger`.

At completion, log an `info` event with:

```text
operation: tag_segs_with_external_llm
segment_count
external_llm_model
seg_image_mode
universal_positive_present
replace_underscore
trailing_comma
exclude_tags_present
```

Provider failures should preserve exception context. Existing external LLM provider errors can propagate if they are already actionable.

Use Comfy progress in the service:

- Start with one initial progress update after validation/model resolution.
- Update after each SEG LLM call.
- Update once after conditioning encode.

## Security And Runtime Constraints

- Use the ComfyUI virtual environment located at `..\..\venv`.
- Do not create a repository-local `.venv`.
- Do not use global Python for verification.
- Use PowerShell command forms on Windows.
- Do not write temporary crop files to disk.
- Do not log API keys, provider secrets, or full data URLs.
- Do not execute workflow-provided code.
- Keep network calls inside the external LLM client/provider boundary.
- Keep filesystem, subprocess, and network behavior out of node classes and domain logic.
- Keep import-time behavior cheap. The node schema may read cached model names, but must not make provider requests during import/schema declaration.

## Required Files To Touch

Expected new files:

```text
simple_syrup/services/tag_segs_with_external_llm_service.py
simple_syrup/nodes_v3/tag_segs_with_external_llm.py
tests/test_external_llm_segs_images.py
tests/test_tag_segs_with_external_llm_service.py
tests/test_tag_segs_with_external_llm_v3_node.py
```

Expected modified files:

```text
simple_syrup/runtime/external_llm_images.py
simple_syrup/services/external_llm_prompt_service.py
simple_syrup/nodes/tooltips.py
simple_syrup/nodes_v3/__init__.py
tests/test_registration.py
tests/test_node_tooltips.py
```

Only modify additional files if implementation proves they are the correct ownership location.

Do not add a file under:

```text
simple_syrup/nodes/
```

for this node.

## Implementation Sequence

1. Add characterization tests around the existing external LLM prompt service if extraction is needed.
2. Add the SEG crop image encoder and image-mode tests.
3. Add response formatting logic and service tests.
4. Add the application service with fake-boundary tests.
5. Add the v3 node schema and execute forwarding tests.
6. Register the v3 node and update registration tests.
7. Add tooltips and update tooltip coverage tests.
8. Run focused tests while developing.
9. Run full verification gates before completion.

## Verification Commands

Run all commands from the repository root with PowerShell syntax.

Required gates:

```powershell
..\..\venv\Scripts\ruff.exe format .
..\..\venv\Scripts\ruff.exe check .
..\..\venv\Scripts\mypy.exe --strict simple_syrup tests
..\..\venv\Scripts\python.exe -m pytest -n auto -q
```

Do not substitute global tools.

If a required tool is missing from `..\..\venv`, install or update development dependencies in that environment before verification. Do not create a local `.venv`.

## Definition Of Done

- The node is exported only through Comfy v3.
- `SimpleSyrup.TagSEGSWithExternalLLM` appears in `get_nodes()`.
- The node returns unchanged SEGS plus one positive conditioning entry per SEG.
- One external LLM call is made per SEG.
- `transparent mask`, `black mask`, and `full crop` are implemented and tested.
- Formatting controls are implemented and tested.
- All validation failures are explicit and actionable.
- Tooltips cover every input and output.
- Tests cover service behavior, image encoding, v3 schema, execution forwarding, registration, and tooltip coverage.
- No legacy node class or legacy mapping export is introduced.
- Required ruff, mypy, and pytest gates pass in `..\..\venv`.
