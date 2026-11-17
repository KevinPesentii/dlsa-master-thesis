#!/usr/bin/env bash
# Run a matrix on a Spot machine that AWS may stop and start again (docs/cloud_runs.md section 18).
# A persistent Spot request with interruption behaviour "Stop" keeps the disk but not the processes,
# so `install` writes a systemd service that starts the matrix now and again at every boot until it
# has finished. run_matrix.py skips finished jobs and each search resumes from its trials.jsonl: an
# interruption costs the trials in flight. At the end it packs the matrix's out dir, the console log,
# the output check (scripts/check_matrix_outputs.py -> runs/<name>_outputs.txt) and every finished
# run directory into ~/<name>.tgz (+ .sha256), leaves ~/<name>.done (later boots then do nothing) and
# shuts the machine down.
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
    # "matrix done in" anywhere after this boot's start line (a long summary pushes it out of any tail)
    start=$(grep -n "start or resume" "$log" | tail -n 1 | cut -d: -f1)
    if [ "$status" -eq 0 ] && tail -n +"$start" "$log" | grep -q "matrix done in"; then
        # every finished run directory travels too (not only runs/_matrix/<name>), and the output check
        "$PY" scripts/check_matrix_outputs.py --matrix "$matrix" "$@" > "runs/${name}_outputs.txt" 2>&1 || true
        "$PY" scripts/check_matrix_outputs.py --matrix "$matrix" "$@" --list-run-dirs > "runs/${name}_rundirs.txt"
        tar -czf "$HOME/$name.tgz" -C "$REPO" "runs/_matrix/$name" "$log" "runs/${name}_outputs.txt" \
            "runs/${name}_rundirs.txt" -T "runs/${name}_rundirs.txt"
        sha256sum "$HOME/$name.tgz" > "$HOME/$name.tgz.sha256"
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
