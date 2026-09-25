# Testing & CI/CD policy

## Suites

| Suite | Path | Marker | Purpose |
|-------|------|--------|---------|
| Unit | `tests/unit/` | `@pytest.mark.unit` | Fast, synthetic data; no dengue CSV required |
| System | `tests/system/` | `@pytest.mark.system` | NMI + vendored `data/` (dengue.csv, data*.txt) end-to-end |

Datasets live in [`data/`](../data/) (copied from [mrvr/NMI](https://github.com/mrvr/NMI)). The NMI *library* still comes from `NMI_ROOT` / sibling `../NMI` (cloned automatically on GitHub Actions).

## On every check-in (GitHub Actions)

Workflow: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

1. **Inspect** — `scripts/inspect_tests.py` verifies both suites exist, tests are marked, and imports still resolve. Stale or empty tests fail the build (update or remove them).
2. **Unit tests** — `pytest -m unit`
3. **System tests** — clone [mrvr/NMI](https://github.com/mrvr/NMI.git), then `pytest -m system`
4. **Release** (push to `main` / `master` only, after green unit + system) — bump patch semver (`VERSION` + `denguecad.__version__`), tag `vX.Y.Z`, create a GitHub Release

## Continuously upgrading tests

Whenever production code under `denguecad/` changes:

1. Update or add **unit** tests for the changed logic.
2. Update **system** tests if behavior visible through NMI / dengue pipeline changed.
3. Remove tests that no longer match the API (the inspector flags broken imports / empty files).
4. Re-run locally:

```bash
source .venv/bin/activate
bash scripts/ci_local.sh
```

## Local commands

```bash
python scripts/inspect_tests.py
python -m pytest -q -m unit
python -m pytest -q -m system
bash scripts/ci_local.sh
```
