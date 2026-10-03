#!/usr/bin/env python3
"""
Render the Mermaid diagrams in docs/ARCHITECTURE.md to PNG and SVG.

Each diagram is a ```mermaid block preceded by ``<!-- diagram: NAME -->``.
Writes docs/architecture/NAME.{mmd,png,svg} using mermaid-cli via npx.
Use --source / --outdir for other Markdown files (e.g. the methods guide
flowcharts in docs/methods/FLOWCHARTS.md).

Requires Node.js (npx) and a Chrome/Chromium binary; set CHROME_PATH to override
auto-detection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ARCHITECTURE.md"
OUTDIR = ROOT / "docs" / "architecture"
MERMAID_CLI = "@mermaid-js/mermaid-cli@11"

BLOCK_RE = re.compile(
    r"<!--\s*diagram:\s*(?P<name>[\w.-]+)\s*-->\s*```mermaid\n(?P<body>.*?)```",
    re.DOTALL,
)

MERMAID_CONFIG = {
    "theme": "base",
    "themeVariables": {
        "fontFamily": "Helvetica, Arial, sans-serif",
        "fontSize": "16px",
        "primaryColor": "#ebf4ff",
        "primaryBorderColor": "#2c5282",
        "primaryTextColor": "#1a202c",
        "lineColor": "#4a5568",
        "clusterBkg": "#f7fafc",
        "clusterBorder": "#a0aec0",
    },
    "flowchart": {"htmlLabels": True, "curve": "basis", "padding": 12},
}


def find_chrome() -> str:
    env = os.environ.get("CHROME_PATH")
    if env:
        return env
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("No Chrome/Chromium found; set CHROME_PATH.")


def extract_diagrams(text: str) -> list[tuple[str, str]]:
    return [(m["name"], m["body"].strip() + "\n") for m in BLOCK_RE.finditer(text)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--scale", type=float, default=3.0, help="PNG pixel scale")
    parser.add_argument("--only", nargs="*", help="Render only these diagram names")
    parser.add_argument("--source", type=Path, default=SOURCE, help="Markdown file with tagged diagrams")
    parser.add_argument("--outdir", type=Path, default=OUTDIR, help="Output directory")
    args = parser.parse_args()
    outdir = args.outdir

    if not shutil.which("npx"):
        sys.exit("npx not found; install Node.js to render Mermaid diagrams.")

    diagrams = extract_diagrams(args.source.read_text(encoding="utf-8"))
    if args.only:
        diagrams = [d for d in diagrams if d[0] in set(args.only)]
    if not diagrams:
        sys.exit(f"No tagged mermaid blocks found in {args.source}")

    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "mermaid.json"
        cfg.write_text(json.dumps(MERMAID_CONFIG))
        pup = tmp_path / "puppeteer.json"
        pup.write_text(json.dumps({"executablePath": find_chrome(), "args": ["--no-sandbox"]}))

        env = {**os.environ, "PUPPETEER_SKIP_DOWNLOAD": "1"}
        for name, body in diagrams:
            src = outdir / f"{name}.mmd"
            src.write_text(body, encoding="utf-8")
            for ext, extra in (("png", ["-s", str(args.scale)]), ("svg", [])):
                out = outdir / f"{name}.{ext}"
                cmd = [
                    "npx", "-y", "-p", MERMAID_CLI, "mmdc",
                    "-i", str(src), "-o", str(out),
                    "-b", "white", "-c", str(cfg), "-p", str(pup), *extra,
                ]
                subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
            print(f"rendered {name}")

    print(f"Wrote {len(diagrams)} diagrams to {outdir}")


if __name__ == "__main__":
    main()
