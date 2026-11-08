#!/usr/bin/env bash
# Run a matrix on a Spot machine that AWS may stop and start again (docs/cloud_runs.md section 18).
# A persistent Spot request with interruption behaviour "Stop" keeps the disk but not the processes,
# so `install` writes a systemd service that starts the matrix now and again at every boot until it
# has finished. run_matrix.py skips finished jobs and each search resumes from its trials.jsonl: an
# interruption costs the trials in flight. At the end it packs the matrix's out dir and console log
# into ~/<name>.tgz, leaves ~/<name>.done (later boots then do nothing) and shuts the machine down.
#   bash scripts/spot_resume.sh install configs/search_t4_k30_matrix.yaml --jobs 46 --threads 2
#   bash scripts/spot_resume.sh status
#   bash scripts/spot_resume.sh unit <matrix> [run_matrix args]   # print the service, install nothing
#   bash scripts/spot_resume.sh run <matrix> [run_matrix args]    # what the service runs
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=${PY:-$HOME/afe-venv/bin/python}
SERVICE=afe-matrix

name_of() {   # the matrix's out dir name under runs/_matrix
    local name
    name=$("$PY" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$1")
    shift
    case " $* " in *" --smoke "*) name=${name}_smoke ;; esac
    echo "$name"
}

unit() {
    local matrix=$1; shift
    cat <<EOF
[Unit]
Description=afe matrix $(name_of "$matrix" "$@"), started at every boot until done
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
Environment=PY=$PY
ExecStart=/bin/bash $REPO/scripts/spot_resume.sh run $matrix $*
Restart=no

[Install]
WantedBy=multi-user.target
EOF
}

case "${1:-}" in
  unit)
    shift; unit "$@" ;;
  install)
    shift
    unit "$@" | sudo tee /etc/systemd/system/$SERVICE.service > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable --now $SERVICE.service
    echo "$SERVICE started, and enabled at every boot; progress: tail -f $REPO/runs/$(name_of "$@")_console.log" ;;
  run)
    matrix=$2; shift 2
    name=$(name_of "$matrix" "$@")
    log=runs/${name}_console.log
    if [ -e "$HOME/$name.done" ]; then echo "$name done already ($HOME/$name.done)"; exit 0; fi
    mkdir -p runs
    echo "=== $(date -u +%FT%TZ) start or resume" >> "$log"
    status=0
    "$PY" scripts/run_matrix.py --matrix "$matrix" "$@" >> "$log" 2>&1 || status=$?
    tail_lines=$(tail -n 40 "$log")
    if [ "$status" -eq 0 ] && [[ $tail_lines == *"matrix done in"* ]]; then
        tar -czf "$HOME/$name.tgz" -C "$REPO" "runs/_matrix/$name" "$log"
        touch "$HOME/$name.done"
        echo "=== $(date -u +%FT%TZ) packed $HOME/$name.tgz, shutting down" >> "$log"
        sudo shutdown -h now
    else
        echo "=== $(date -u +%FT%TZ) run_matrix ended with status $status before the end: not packed" >> "$log"
    fi ;;
  status)
    systemctl --no-pager status $SERVICE.service 2>/dev/null | head -n 4 || true
    ls -1 "$HOME"/*.done 2>/dev/null || echo "no matrix done yet" ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
