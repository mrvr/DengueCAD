"""
Bridge to the sibling NMI project (https://github.com/mrvr/NMI).

Resolves ``nmilib`` from a local checkout so DengueCAD can call
``non_parametric_imputation`` without installing NMI as a package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Default: sibling folder next to DengueCAD
_DEFAULT_NMI_ROOT = Path(__file__).resolve().parents[2] / "NMI"


def resolve_nmi_root(nmi_root: Optional[str | Path] = None) -> Path:
    """
    Locate the NMI project root.

    Order:
      1. Explicit ``nmi_root`` argument
      2. Environment variable ``NMI_ROOT``
      3. Sibling directory ``../NMI`` relative to DengueCAD
    """
    if nmi_root is not None:
        root = Path(nmi_root).expanduser().resolve()
    elif os.environ.get("NMI_ROOT"):
        root = Path(os.environ["NMI_ROOT"]).expanduser().resolve()
    else:
        root = _DEFAULT_NMI_ROOT.resolve()

    if not (root / "nmilib.py").is_file():
        raise FileNotFoundError(
            f"NMI library not found at {root}. "
            "Clone https://github.com/mrvr/NMI or set NMI_ROOT / pass nmi_root=."
        )
    return root


def ensure_nmi_on_path(nmi_root: Optional[str | Path] = None) -> Path:
    """Insert the NMI project directory on ``sys.path`` and return its path."""
    root = resolve_nmi_root(nmi_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def import_nmilib(nmi_root: Optional[str | Path] = None):
    """Import and return the ``nmilib`` module from the local NMI checkout."""
    ensure_nmi_on_path(nmi_root)
    import nmilib  # type: ignore

    return nmilib
