#!/usr/bin/env python3
"""
Inspect unit and system tests for structural validity.

Run on every CI check-in. Fails when the suite is empty, unmarked, uncollectable,
or references modules that no longer exist. Prints a short hygiene report so
maintainers can update or remove stale tests.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "tests" / "unit"
SYSTEM = ROOT / "tests" / "system"
PKG = ROOT / "denguecad"


def _test_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("test_*.py"))


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(alias.name for alias in node.names)
    return mods


def _package_modules() -> set[str]:
    names = {"denguecad"}
    for py in PKG.glob("*.py"):
        if py.name == "__init__.py":
            continue
        names.add(f"denguecad.{py.stem}")
    return names


def _has_pytestmark(path: Path, expected: str) -> bool:
    src = path.read_text(encoding="utf-8")
    return f'pytest.mark.{expected}' in src or f"pytest.mark.{expected}" in src


def _count_test_functions(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def inspect_suite(name: str, folder: Path, marker: str) -> list[str]:
    errors: list[str] = []
    if not folder.is_dir():
        return [f"{name}: missing directory {folder}"]

    files = _test_files(folder)
    if not files:
        errors.append(f"{name}: no test_*.py files under {folder}")
        return errors

    pkg_mods = _package_modules()
    total_tests = 0
    print(f"\n[{name}] {len(files)} file(s) in {folder.relative_to(ROOT)}")
    for path in files:
        n = _count_test_functions(path)
        total_tests += n
        marked = _has_pytestmark(path, marker)
        status = "ok" if n and marked else "NEEDS UPDATE"
        print(f"  - {path.relative_to(ROOT)}: {n} tests, marker={marker}? {marked} [{status}]")
        if n == 0:
            errors.append(f"{path}: contains no test_* functions (remove or rewrite)")
        if not marked:
            errors.append(f"{path}: missing pytestmark = pytest.mark.{marker}")

        for mod in _collect_imports(path):
            if not mod.startswith("denguecad"):
                continue
            # Allow denguecad and denguecad.submodule
            root_mod = ".".join(mod.split(".")[:2]) if mod.count(".") else mod
            if root_mod not in pkg_mods and mod not in pkg_mods and mod != "denguecad":
                # Only flag clear missing submodules like denguecad.foo
                parts = mod.split(".")
                if len(parts) >= 2 and parts[0] == "denguecad":
                    candidate = PKG / f"{parts[1]}.py"
                    if not candidate.is_file() and parts[1] != "__init__":
                        errors.append(
                            f"{path}: imports missing module {mod} — update or remove test"
                        )

    if total_tests == 0:
        errors.append(f"{name}: zero test functions collected")
    else:
        print(f"  total {name} tests: {total_tests}")
    return errors


def main() -> int:
    print("DengueCAD test inspector")
    print(f"root: {ROOT}")
    errors: list[str] = []
    errors.extend(inspect_suite("unit", UNIT, "unit"))
    errors.extend(inspect_suite("system", SYSTEM, "system"))

    # Legacy flat tests should not linger
    legacy = list((ROOT / "tests").glob("test_*.py"))
    if legacy:
        for path in legacy:
            errors.append(
                f"legacy flat test {path.relative_to(ROOT)} — move to tests/unit or tests/system"
            )

    if errors:
        print("\nINVALID / STALE TESTS:")
        for err in errors:
            print(f"  × {err}")
        return 1

    print("\nAll unit and system tests look structurally valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
