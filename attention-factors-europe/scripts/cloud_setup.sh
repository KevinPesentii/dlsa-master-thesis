#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 machine for the runners (docs/cloud_runs.md).
# Pins the versions of Henrik's afe env of 2026-09-25 so cloud and laptop runs match.
#   bash scripts/cloud_setup.sh          # CPU machine: the CPU build of torch
#   bash scripts/cloud_setup.sh --gpu    # GPU machine (section 19): the same torch built with CUDA, from PyPI,
#                                        # which brings its own CUDA libraries; needs only the NVIDIA driver
set -euo pipefail
cd "$(dirname "$0")/.."

sudo apt-get update -q
sudo apt-get install -y -q python3-venv tmux htop
python3 -m venv ~/afe-venv
~/afe-venv/bin/pip install -q --upgrade pip
if [ "${1:-}" = "--gpu" ]; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
        || { echo "no NVIDIA driver: use a GPU AMI with the driver (section 19)"; exit 1; }
    ~/afe-venv/bin/pip install -q torch==2.14.0 pytest      # pytest: the CUDA tests run on the machine
else
    ~/afe-venv/bin/pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
fi
~/afe-venv/bin/pip install -q numpy==2.5.3 pandas==2.3.3 pyarrow==25.0.1 scipy==1.18.1 pyyaml==6.0.3
~/afe-venv/bin/pip install -q --no-deps -e .
~/afe-venv/bin/python -c "import afe, torch; print('afe ok, torch', torch.__version__, 'cpus', torch.get_num_threads())"
if [ "${1:-}" = "--gpu" ]; then
    ~/afe-venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; \
print('cuda', torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
fi
