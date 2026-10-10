"""INDEX.md of a thesis results folder: every table and figure with title, caption, notes,
files and source, from tables/index.json and figures/index.json.

    python scripts/thesis_handoff.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = ["# Index of tables and figures", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} from commit {commit} (scripts/thesis_tables.py, "
             "scripts/thesis_figures.py). Numbers: CSV files at full precision; .md/.tex formatted.", ""]
    for kind in ("tables", "figures"):
        idx = json.loads((args.out / kind / "index.json").read_text(encoding="utf-8"))
        lines += [f"## {kind.capitalize()}", ""]
        for e in idx:
            lines += [f"### {e['title']}", "", e["caption"], ""]
            lines += [f"- Note: {n}" for n in e.get("notes", [])]
            lines += [f"- Files: " + ", ".join(f"`{f}`" for f in e["files"]), f"- Source: {e['source']}", ""]
    (args.out / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out / 'INDEX.md'}")


if __name__ == "__main__":
    main()
