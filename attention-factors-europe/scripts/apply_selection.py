"""Write a finished search's selected values into configs, keeping their comments.

    python scripts/apply_selection.py runs/<search run dir> configs/us_headline.yaml configs/europe_headline.yaml

The values are those of every searched or fixed key in the selected config (base + the
winner's overrides; data / sample keys excluded), so a winning base candidate is carried
as well. Each `section.key` line is replaced in place; a key a config does not have is an
error. A first line `# SELECTED: ...` records the search run, the candidate and its
validation net Sharpe.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def selected_values(search_dir: Path) -> tuple[dict, dict]:
    """(winner, values) of a finished search: the value of EVERY searched or fixed key in the
    selected config (base config + winner's overrides), so a winning "base" candidate carries
    the base config's values too; data / sample keys of a fixed block are never carried."""
    man = json.loads((search_dir / "manifest.json").read_text())["config"]
    S, base = man["search"], man["base"]
    w = json.loads((search_dir / "metrics.json").read_text())["winner"]
    keys = [*(S.get("fixed") or {}), *S["search"].get("space", {}), *S["search"].get("grid", {})]
    values = {}
    for k in keys:
        section, name = k.split(".")
        if section not in ("data", "sample"):
            values[k] = w["overrides"].get(k, base[section][name])
    return w, values


def scalar(v) -> str:
    return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()


def set_value(text: str, key: str, value) -> str:
    section, name = key.split(".")
    m = re.search(rf"^{re.escape(section)}:[^\n]*\n((?:[ \t]+[^\n]*\n|[ \t]*\n)*)", text, flags=re.M)
    if not m:
        raise KeyError(f"no section {section}")
    block = m.group(1)
    pat = re.compile(rf"^(  {re.escape(name)}:[ \t]*)([^#\n]*?)([ \t]*(#[^\n]*)?)$", flags=re.M)
    if len(pat.findall(block)) != 1:
        raise KeyError(f"{key}: not exactly one line in the config")
    new = pat.sub(lambda g: g.group(1) + scalar(value) + g.group(3), block)
    return text[:m.start(1)] + new + text[m.end(1):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("search_dir", type=Path)
    ap.add_argument("configs", nargs="+", type=Path)
    args = ap.parse_args()
    S = json.loads((args.search_dir / "manifest.json").read_text())["config"]["search"]
    w, values = selected_values(args.search_dir)
    tag = (f"# SELECTED: {args.search_dir.name} ({S.get('base_config', '?')} search), candidate {w['name']}, "
           f"validation net SR {w['val_net_SR']:.2f} over seeds {w.get('n_seeds', '?')}\n")
    for path in args.configs:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^# SELECTED: [^\n]*\n", "", text)
        for k, v in values.items():
            text = set_value(text, k, v)
        path.write_text(tag + text, encoding="utf-8", newline="")
        print(f"{path}: {values}")


if __name__ == "__main__":
    main()
