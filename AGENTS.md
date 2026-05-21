# AGENTS.md

## Mission Statement

This project provides a high-quality Python ComfyUI custom node extension pack.

Engineering priority is strict architecture, strong separation of concerns, complete feature integrations, behavior safety during structural change, deterministic node behavior, explicit validation, ComfyUI compatibility, runtime safety, and long-term maintainability.

## Purpose

- This file defines engineering guardrails for this repository.
- This file governs architecture, code quality, typing, testing, observability, runtime safety, ComfyUI node behavior, workflow behavior, and extension packaging.
- Do not use this file for feature specs or product planning.

## Behavior Boundary

- Preserve existing user-facing behavior unless explicitly approved to change.
- Preserve compatibility for supported ComfyUI node classes, node names, input definitions, output definitions, categories, return types, and execution behavior unless explicitly approved to change.
- Preserve compatibility for serialized ComfyUI workflows that reference this node pack unless explicitly approved to change.
- Treat current node signatures, widget names, return shapes, validation behavior, and workflow-facing identifiers as the contract.
- Change internals freely within that boundary.

## Environment and Gate Execution

- All verification commands must run against the ComfyUI virtual environment located two directories above this repository.
- From this repository root, the required Python environment is `..\..\venv`.
- Do not create or use a repository-local `.venv`.
- Do not run quality gates with global/system Python.
- If the ComfyUI virtual environment is missing or stale, pause and ask for maintainer direction before recreating it.
- If development tools are missing from the ComfyUI virtual environment, install or update this node pack and its development tooling into that environment.
- Run all commands from the repository root.
- Use PowerShell syntax and Windows paths.
- The ComfyUI virtual environment is authoritative because node behavior may depend on the exact installed ComfyUI, PyTorch, image, tensor, and custom node runtime dependencies available to the host application.

### Required Command Forms

- Tests: `..\..\venv\Scripts\python.exe -m pytest -n auto -q`
- Lint: `..\..\venv\Scripts\ruff.exe check .`
- Format: `..\..\venv\Scripts\ruff.exe format .`
- Type check: `..\..\venv\Scripts\mypy.exe --strict simple_syrup tests`

If a required tool is missing from `..\..\venv`, install or update development dependencies in that environment before verification. Do not substitute global tools.

## Core Engineering Principles

- Use strict object-oriented design where ownership, state, lifecycle, or collaboration boundaries exist.
- Enforce strong separation of concerns as the primary architecture objective.
- Keep modules cohesive and boundaries explicit.
- Assign one authoritative owner per concern.
- Other components may participate in a concern only by using the authoritative owner. They must not re-implement that concern in parallel.
- Reassess ownership before extending an existing structure.
- If a change introduces a distinct responsibility, change cadence, or collaboration boundary, split or extract it as part of the change.
- Complete refactors fully. Update all callsites, remove dead code, remove temporary bridges, and make the new design native to the codebase.
- Complete feature additions fully. Wire the feature through the relevant node class, service layer, validation path, tests, typing, and ComfyUI registration path required by the behavior.
- Do not leave partial implementations, unused code paths, TODO-driven behavior, or follow-up cleanup inside the completed change.
- Do not add internal compatibility layers, internal shims, dual internal paths, legacy fallbacks, or transitional adapters.
- Preserve compatibility only at public ComfyUI-facing or persisted workflow boundaries when required by the behavior contract.
- Favor DRY when it reduces repeated change risk.
- Avoid abstractions that hide intent.

## Architecture Rules

- Organize code into clear layers with one-way dependencies.
- ComfyUI integration layer: `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, node categories, input/output declarations, and ComfyUI import-time registration.
- Node API layer: thin node classes exposing ComfyUI-facing methods such as `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, and execution entry points.
- Application/service layer: orchestration for node behavior, validation flow, and feature-level use cases.
- Domain layer: stable internal models, value objects, policies, and pure behavior.
- Runtime/adapter layer: filesystem access, image/audio/model IO, ComfyUI object adaptation, subprocess boundaries, network boundaries, and optional external integrations.
- Shared layer: small cross-layer primitives with no higher-level dependencies.
- Higher-level layers may depend on lower-level layers.
- Lower-level layers must not depend on higher-level layers.
- ComfyUI registration must stay thin and must not contain feature logic.
- Node classes must delegate non-trivial behavior to application services or domain objects.
- Domain logic must not import ComfyUI modules.
- Runtime adapters own external system interaction.
- Keep filesystem, subprocess, and network details out of node classes and domain logic.
- Place code by ownership and dependency direction, not convenience or proximity.
- Avoid god classes and monolithic files.
- Split by responsibility, not convenience.

## Structural Change Rules

- For behavior-critical areas, work in two steps:
  1. Add characterization/regression tests for existing behavior.
  2. Perform structural changes behind those tests.
- Behavior-critical areas include node registration, `INPUT_TYPES`, `RETURN_TYPES`, widget names, output ordering, execution return shapes, workflow compatibility, validation behavior, file IO, model IO, image/audio tensor handling, and ComfyUI import behavior.
- Do not start structural changes in an area without behavior safeguards for that area.
- When behavior spans multiple components, trace the current ownership and data flow before editing.
- Correct the ownership model instead of layering compensating patches across consumers.
- Land structural changes as complete vertical slices.
- Do not land large unverified rewrites.
- If behavior changes are intentional, explicitly call them out and test them as new behavior.
- Current module layout does not constrain improvement.
- Reorganize modules when it improves architecture.
- Align touched modules with the ownership and dependency rules in this file.

## ComfyUI Node Rules

- Public node identifiers are compatibility-sensitive.
- Do not rename node classes, display names, categories, input keys, output names, return types, or function names without explicit approval.
- Keep ComfyUI-facing node classes small and predictable.
- `INPUT_TYPES` must be deterministic and must not perform expensive IO.
- Importing the node pack must not perform heavy computation, network access, model loading, or destructive filesystem operations.
- Node execution must validate inputs before performing side effects.
- Node execution must return exactly the declared output shape.
- Optional behavior must be explicit in inputs and tested.
- Hidden inputs and ComfyUI-specific metadata must be documented in code through clear names and docstrings.
- Any workflow-affecting default value change is a behavior change and requires tests.
- Errors surfaced to ComfyUI users must be actionable.

## Tooltip Standards

- Every ComfyUI-facing node must provide a concise, user-facing `DESCRIPTION`.
- Every visible node input, optional input, hidden input, and output tooltip supported by the API must have tooltip text.
- Tooltips should help a workflow builder make the right choice without reading code.
- Tooltips must be concise, layman-friendly, and technically accurate enough to explain the practical effect of the setting.
- Prefer one sentence. Use two only when the field has an important tradeoff or safety implication.
- Explain what the control does, what changing it affects, and any important tradeoff.
- Include units when relevant, such as pixels, latent pixels, scale multiplier, mask value, batch count, or strength.
- Explain directionality when useful: higher/lower, enabled/disabled, larger/smaller.
- Avoid implementation jargon unless the user needs it to make a good workflow decision.
- Do not repeat the field name as a definition.
- Do not document removed behavior, imagined alternatives, or choices the product does not expose.
- Keep legacy `INPUT_TYPES` tooltips and Comfy v3 schema tooltips aligned when both export paths expose the same node or field.

## ComfyUI Node Export Rules

- When adding, renaming, or removing a ComfyUI node, update and verify every export path used by this repository.
- Legacy ComfyUI mapping exports must be updated in `simple_syrup/nodes/__init__.py`:
  - `NODE_CLASS_MAPPINGS`
  - `NODE_DISPLAY_NAME_MAPPINGS`
  - `__all__`
- Comfy v3 entrypoint exports must be updated when the node should be visible through the v3 API:
  - `simple_syrup/nodes_v3/__init__.py`
  - `get_nodes()`
  - a v3 wrapper class when needed
- The root package export in repository root `__init__.py` must continue exposing the relevant mappings and `comfy_entrypoint`.
- Tests must cover every export path used by the node:
  - A registration test must assert the node id and display name exist in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
  - A v3 entrypoint test must assert `comfy_entrypoint().get_node_list()` includes the node when it is expected to be visible through Comfy v3.
  - If a v3 node is conditional, tests must cover both the available and unavailable conditions and prove unrelated v3 nodes remain exported.
- Do not consider a node addition complete from `NODE_CLASS_MAPPINGS` alone. A node is not fully exported until every repository-supported ComfyUI export path is updated and tested.

## Code Organization and Readability

- Write self-documenting code with expressive, concise names.
- Place new code deliberately in the module where it naturally belongs.
- Keep files intentionally organized so reading order reflects design intent.
- Do not place code opportunistically "where it works".
- Remove obsolete code paths when replacements are complete.
- Keep ComfyUI registration concerns at the package boundary.
- Keep node declaration concerns in node modules.
- Keep reusable behavior in services or domain modules.
- Keep filesystem, network, subprocess, and external-library adaptation in runtime adapter modules.

## Docstrings and Comments

- Docstrings are mandatory for all new and changed modules, classes, functions, and methods.
- Use concise imperative docstrings for simple logic.
- Use Google-style docstrings for complex logic.
- Docstrings must explain rationale, constraints, and intent.
- Docstrings must not restate obvious mechanics.
- Inline comments are allowed only for non-obvious behavior, invariants, edge cases, or external constraints.

## Documentation Policy

- Do not create new docs files, README variants, design docs, ADRs, roadmap files, or notes unless explicitly requested by the maintainer.
- Required context must live in code, type hints, tests, and docstrings.
- Documentation and explanatory writing must describe the product directly as it exists now.
- Do not document against removed features, imagined alternatives, or non-existent choices.

## Typing Policy

- Strong typing is required for all new code.
- Modified code must be typed as part of the change.
- Type hints are mandatory on function signatures and key internal state.
- Use explicit domain types, dataclasses, TypedDicts, Protocols, and type narrowing instead of `Any`.
- `Any` is allowed only at ComfyUI, JSON, tensor, or dynamic plugin boundaries and must be narrowed before core logic relies on it.
- Run `mypy --strict` for type verification.
- Temporary typing relaxations are allowed only when explicitly justified inline and tracked for removal.

## Logging, Errors, and Observability

- Observability is mandatory.
- Use structured, actionable logging with context identifiers where relevant.
- Include enough context to diagnose failures quickly, such as node class, display name, input name, output name, file path, model name, operation, and ComfyUI workflow-facing identifier.
- Use log levels consistently: `debug`, `info`, `warning`, `error`.
- Preserve exception context and stack traces for unexpected failures.
- `print` is not allowed for runtime diagnostics.
- Bare `except:` is not allowed.
- `except Exception` must be narrow, intentional, and log context plus failure reason.
- Silent exception swallowing is not allowed.
- Errors exposed from node, service, validation, and runtime boundaries must be explicit and actionable.

## Desktop Security and Safety Rules

- Treat workflow inputs, filesystem paths, model paths, image/audio paths, generated data, subprocess execution, and network access as security-sensitive.
- Never execute untrusted code paths from workflows, metadata, file contents, or generated data.
- Validate and sanitize external paths and user-provided file references.
- Use structured parsing and validation for JSON, metadata, and configuration.
- Use subprocess argument lists, never shell-string execution.
- Set explicit timeouts for network operations.
- Fail closed when trust, schema validation, path validation, or version validation is uncertain.
- Never log secrets, tokens, credentials, or sensitive local paths beyond what is necessary for diagnosis.
- Do not silently continue after invalid node inputs, missing files, invalid model references, failed conversions, malformed metadata, or unsupported data shapes.

## Testing Policy

- Add or update tests for every behavior change and every bug fix.
- Add characterization tests before structural changes to behavior-critical areas.
- New behavior must not be unverified.
- Include success and failure path coverage.
- Include regression tests for fixed bugs.
- Keep tests deterministic and isolated.
- Use real behavior tests over excessive mocking.
- Mock only external boundaries such as ComfyUI runtime calls, filesystem errors, network calls, subprocesses, random generation, and time.
- Node behavior must be tested at the narrowest useful level and through integration-style tests when ComfyUI-facing shape matters.
- Node registration changes require tests for exported mappings.
- Tooltip coverage must be tested for exported node descriptions, inputs, and outputs supported by each ComfyUI API path.
- Input/output signature changes require workflow-facing compatibility tests.
- Runtime behavior requires tests for success and failure paths.
- Add or update frontend tests for frontend behavior changes.
- Settings UI changes require tests for registration, default value, persistence call, and failure handling.
- Backend/frontend settings bridges require Python tests for routes and persistence plus TypeScript tests for API calls.

## Test Execution Rules

- Run tests in parallel using xdist.
- Default command: `..\..\venv\Scripts\python.exe -m pytest -n auto -q`.
- If running a focused subset during development, run the full suite before completion.
- Failing tests are blocking.

## Python Toolchain

- Formatter: `ruff format`
- Linter: `ruff check`
- Type checker: `mypy --strict`
- Test runner: `pytest -n auto -q`

## TypeScript / Frontend Toolchain

- Frontend code lives under `web/src`.
- Frontend tests live under `web/tests`.
- Built browser artifacts live under `web/dist`.
- Do not hand-edit generated files in `web/dist`; edit `web/src` and run the build.
- Use TypeScript for ComfyUI frontend extensions.
- Do not add untyped JavaScript for new frontend behavior unless it is generated build output.
- Keep frontend code small, typed, and framework-free unless explicitly approved.
- Frontend code must not own backend policy. It may display and edit settings, but Python runtime settings remain the source of truth.
- If `package.json` exists, run these from repository root:

```powershell
npm ci
npm run lint:web
npm run typecheck:web
npm run test:web
npm run build:web
```

- Do not use global TypeScript, ESLint, or Vitest directly.

## ComfyUI Frontend Rules

- ComfyUI frontend extensions must be registered from TypeScript source under `web/src`.
- Settings-panel behavior must use Comfy's frontend settings API.
- Browser settings are not authoritative for backend behavior.
- Any frontend setting that affects backend node declarations or execution must be mirrored through an explicit backend route or persisted backend settings file.
- Backend code must validate settings read from disk or HTTP.
- Frontend code must handle backend route failures visibly and conservatively.
- Do not perform model discovery, filesystem assumptions, or download decisions in frontend code.
- Do not duplicate model-listing policy in frontend code; call backend APIs or edit backend settings only.

## Verification Workflow

- Run focused checks continuously while implementing.
- Verify the specific reported behavior directly when feasible.
- Do not declare a node, workflow, runtime, validation, or packaging issue fixed from code inspection alone when a direct test is feasible.
- Run full gates before reporting completion.
- Distinguish observed results from inferred results in updates and completion reports.
- Do not introduce new lint/type failures in modified files.
- Do not report completion if any blocking gate fails.
- If a gate is intentionally deferred, explicitly state the reason and risk.

## Definition of Done

Per change, all of the following are required:

- Behavior is safeguarded by tests.
- New/modified code follows architecture boundaries.
- New/modified code placement reflects ownership and dependency rules in this file.
- Refactors are complete, with callsites updated and obsolete internal paths removed.
- Features are complete, with node registration, node API, services, runtime adapters, validation, typing, and tests updated wherever the behavior requires them.
- New/modified code is typed.
- Required docstrings are present and meaningful.
- Logging/error handling is actionable.
- Security-sensitive boundaries validate inputs and fail closed.
- `..\..\venv\Scripts\ruff.exe format .` passes.
- `..\..\venv\Scripts\ruff.exe check .` passes.
- `..\..\venv\Scripts\mypy.exe --strict simple_syrup tests` passes for the enforced scope.
- `..\..\venv\Scripts\python.exe -m pytest -n auto -q` passes.
- Frontend source is typed and tested when touched.
- Generated frontend artifacts are rebuilt from source.
- `npm run lint:web`, `npm run typecheck:web`, `npm run test:web`, and `npm run build:web` pass when frontend code exists or is touched.
- New, renamed, or removed nodes are updated in all legacy and Comfy v3 export paths, with tests proving both paths expose the intended node set.

## Commit Policy

- Use Conventional Commits: `type(scope): subject`.
- Allowed types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`, `build`, `ci`.
- Keep commits atomic and cohesive.
- Breaking structural changes must be clearly labeled.

## Maintainer Authority

- Maintainer instructions override this file.
- If constraints conflict, pause and ask for maintainer direction before proceeding.
