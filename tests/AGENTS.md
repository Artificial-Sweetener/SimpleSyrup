# AGENTS.md

## Scope

This file supplements the repository-root `AGENTS.md` for every file below
`tests/`.

## Mission

The test suite provides fast, deterministic, behaviorally meaningful evidence
for SimpleSyrup's ComfyUI nodes, domain behavior, services, runtime adapters,
tooling, and persisted workflow contracts.

## Ownership And Placement

- Organize tests by product capability or authoritative behavior owner.
- Do not add test modules directly under `tests/`; the root is reserved for
  pytest configuration and execution policy.
- Within a large capability, split tests by domain, service, runtime, node, or
  integration boundary when those owners change independently.
- Keep capability-specific fixtures, fakes, values, and harnesses in that
  capability's `support/` package.
- Promote support to `tests/support/` only when independent capabilities use
  the same stable testing contract.
- Do not create generic helper, common, misc, or utility dumping grounds.
- Keep `web/tests/` organized by the same capabilities as `web/src/`.
- Update policy entries, runner inventories, imports, and focused-test paths in
  the same change as any test move.

## Behavioral Proof

- Test observable behavior through its authoritative owner.
- Prefer the lightest real component that proves the complete contract.
- Do not mock the behavior under test or duplicate production rules in expected
  result implementations.
- Cover relevant success, failure, boundary, cancellation, cleanup, and
  regression paths.
- System and live-Comfy tests must prove composition that focused owner tests
  cannot prove.

## Parallelism And Isolation

- Tests are parallel-safe by default.
- Do not branch behavior on `PYTEST_XDIST_WORKER`.
- Do not serialize tests to hide leaked state, fixed resources, nondeterminism,
  or unsafe cleanup.
- Use fresh-process isolation only for a demonstrated process-lifetime
  constraint.
- Use serial execution only for an exact global or external resource that
  cannot safely overlap.
- Record every isolated or serial module in `tests/ci_test_policy.py` with a
  reviewed governance disposition.

## Determinism

- Control clocks, timers, randomness, environment, filesystems, subprocesses,
  network responses, and external Comfy state when they affect behavior.
- Do not use arbitrary sleeps as completion conditions.
- Wait for observable state with bounded diagnostic timeouts.
- Do not use retries, skips, weakened assertions, or increased delays to hide
  flakes.
- Every subprocess and network operation must have an explicit failure bound
  and guaranteed cleanup.

## Test State

- Use `tmp_path` and `pathlib.Path` for filesystem behavior.
- Do not write test artifacts into the repository unless the artifact path is
  itself the contract under test.
- Restore environment variables, module replacements, registries, logging
  handlers, working directories, and global settings.
- Keep autouse fixtures limited to universal safety and cleanup invariants.
- Give every fixture one cohesive lifecycle and explicit typed result.

## Typing And Maintainability

- Type tests, fixtures, fakes, builders, and harness APIs.
- Do not add test-only `.pyi` files that shadow executable modules.
- Use explicit protocols or focused typed fakes at dynamic boundaries.
- Keep setup and assertions readable at the test callsite.
- Shared abstractions must reduce repeated change risk without hiding the
  behavior being proved.

## Governance

- Run `..\..\venv\Scripts\python.exe -m tools.check_test_governance`
  after changing test placement, timing, isolation, resources, or execution
  policy.
- Every discovered candidate requires source-level review.
- A classification waiver records legitimate intentional behavior.
- Inappropriate current design requires debt plus an exact remediation waiver.
- Reviewed state must remain fingerprinted, expiring, and file-specific.

## Verification

- Run focused tests continuously while changing a capability.
- Run collection after moving tests.
- Run the full parallel suite before completion.
- Run architecture governance, test governance, formatting, lint, strict
  typing, and frontend gates when applicable.
