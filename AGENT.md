# AGENT

## Purpose

Operational guide for contributors and coding agents working on `voxalign`.

## Working Agreement

1. Complete one milestone at a time.
2. Run quality checks before each commit:
   - `uv run pre-commit run --all-files`
   - `uv run pytest -q`
3. Push every completed milestone to remote branch immediately.
4. Keep changes small, testable, and reversible.

## Tooling Standard

- Runtime management: `mise` (`.mise.toml`)
- Dependency management and execution: `uv`
- Hooks: `pre-commit` and `pre-push`

## Development Commands

```bash
mise install
uv sync --dev --frozen
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

## Milestone Workflow

1. Pick one milestone from `PROJECT.md`.
2. Implement with tests.
3. Run checks.
4. Commit with a focused message.
5. Push to remote.
6. Update status in `PROJECT.md` and/or `docs/implementation-plan.md`.
