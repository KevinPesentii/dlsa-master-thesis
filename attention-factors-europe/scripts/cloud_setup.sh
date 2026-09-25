#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 machine for the runners (docs/cloud_runs.md).
# Pins the versions of Henrik's afe env of 2026-09-25 so cloud and laptop runs match.
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -q
sudo apt-get install -y -q python3-venv tmux htop
python3 -m venv ~/afe-venv
~/afe-venv/bin/pip install -q --upgrade pip
~/afe-venv/bin/pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
~/afe-venv/bin/pip install -q numpy==2.5.3 pandas==2.3.3 pyarrow==25.0.1 scipy==1.18.1 pyyaml==6.0.3
~/afe-venv/bin/pip install -q --no-deps -e .
~/afe-venv/bin/python -c "import afe, torch; print('afe ok, torch', torch.__version__, 'cpus', torch.get_num_threads())"
