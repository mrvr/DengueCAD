"""
Optional locator for a sibling NMI checkout.

NMI imputation does not use that library. Every imputation call goes to
``denguecad.nmi_paper.impute_nmi`` (IEEE TITB 2012).
"""

from __future__ import annotations

import os
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


# def ensure_nmi_on_path(nmi_root: Optional[str | Path] = None) -> Path:
#     """Bypassed. The sibling nmilib is not placed on sys.path."""
#     root = resolve_nmi_root(nmi_root)
#     root_str = str(root)
#     if root_str not in sys.path:
#         sys.path.insert(0, root_str)
#     return root
#
#
# def import_nmilib(nmi_root: Optional[str | Path] = None):
#     """Bypassed. Call denguecad.nmi_paper.impute_nmi (2012) instead."""
#     ensure_nmi_on_path(nmi_root)
#     import nmilib  # type: ignore
#     return nmilib
