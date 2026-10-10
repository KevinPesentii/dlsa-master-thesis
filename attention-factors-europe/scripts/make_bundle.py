"""Pack what the thesis matrix needs for an external machine (docs/cloud_runs.md, section 7).

    python scripts/make_bundle.py [--out runs/_bundle/<stamp>] [--git-bundle]

The code comes from the public fork (git clone --depth 1 -b <branch>), so only the data
travels. The tree must be clean and the branch pushed. Writes:
  COMMIT          branch and commit to clone; the machine checks `git rev-parse HEAD`.
  afe.bundle      only with --git-bundle (a machine without GitHub access): the branch
                  with its whole history, ~0.8 GB.
  afe_data.tar    the schema tables the runners read, under data/ as in the repo. The PCA
                  stage ones are NOT included: the matrix rebuilds them (prep jobs), which
                  is faster than moving 6 GB.
  SHA256SUMS      sha256sum format; on the machine: (cd <repo dir> && sha256sum -c SHA256SUMS)
Licensed CRSP / Compustat data: private machine, encrypted disk, delete it afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from afe import runs  # noqa: E402

TABLES = ("returns", "universe", "features")
FILES = (
    [f"data/us/shared/{t}.parquet" for t in TABLES]
    + [f"data/us/shared/raw/{f}.parquet" for f in ("ff_daily", "ff_monthly", "ff5_daily", "ff5_monthly")]
    + ["data/us/shared/raw/crsp_daily/1989.parquet"]          # the first 252-day PCA window
    + [f"data/europe17/holfill/{t}.parquet" for t in TABLES]  # headline Europe
    + [f"data/europe17/shared/{t}.parquet" for t in TABLES + ("rf_daily", "market_daily", "factors_daily",
                                                              "fx_to_numeraire_daily")]
    + [f"data/europe12/holfill/{t}.parquet" for t in TABLES]  # robustness
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--git-bundle", action="store_true", help="also write the branch as a git bundle")
    args = ap.parse_args()
    commit = runs.git_commit(ROOT)
    if commit.endswith("-dirty"):
        raise SystemExit(f"git tree is dirty ({commit}); commit first")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    out = args.out or ROOT / "runs" / "_bundle" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")

    upstream = subprocess.run(["git", "rev-parse", "@{u}"], cwd=ROOT, text=True, capture_output=True).stdout.strip()
    if upstream != commit:
        print(f"WARNING: {branch} at {commit[:10]} is not what origin has ({upstream[:10] or 'no upstream'}); push it")
    (out / "COMMIT").write_text(f"{branch} {commit}\n", newline="\n")
    if args.git_bundle:
        subprocess.run(["git", "bundle", "create", str(out / "afe.bundle"), branch], cwd=ROOT, check=True)
    sums = []
    with tarfile.open(out / "afe_data.tar", "w") as tar:
        for f in FILES:
            tar.add(ROOT / f, arcname=f)
            sums.append(f"{sha256(ROOT / f)}  {f}")
            print(f"  {f}  {(ROOT / f).stat().st_size / 2 ** 20:7.1f} MB", flush=True)
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", newline="\n")
    size = sum(p.stat().st_size for p in out.iterdir()) / 2 ** 20
    print(f"branch {branch} at {commit[:10]}; {len(FILES)} data files; {size:.0f} MB in {out}")


if __name__ == "__main__":
    main()
