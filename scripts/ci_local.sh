#!/usr/bin/env bash
# Run the same gates as GitHub Actions CI locally (inside .venv).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv — create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo "==> Inspect tests"
"$PY" scripts/inspect_tests.py

echo "==> Unit tests"
"$PY" -m pytest -q -m unit

echo "==> System tests"
"$PY" -m pytest -q -m system

echo "CI local gates passed."
