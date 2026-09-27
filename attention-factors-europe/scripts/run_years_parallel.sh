#!/usr/bin/env bash
# Run a runner once per out-of-sample year (and seed), JOBS processes at a time, then merge
# the per-year run directories with scripts/merge_year_runs.py. For many-core machines;
# see docs/cloud_runs.md.
#
#   scripts/run_years_parallel.sh [--jobs 24] [--threads 4] [--first 1998] [--last 2021]
#                                 [--seeds "0"] -- <runner.py> [runner args, without --years/--seed/--threads]
#
#   scripts/run_years_parallel.sh --jobs 24 --threads 4 -- scripts/run_attention_us_val.py --K 30
#
# Logs go to runs/_logs/<stamp>/s<seed>_<year>.log. Launches are 2 s apart to spread the
# data loading; run directory name clashes are resolved in runs.create_run.

set -uo pipefail
cd "$(dirname "$0")/.."

JOBS=24; THREADS=4; FIRST=1998; LAST=2021; SEEDS="0"
PYTHON="${PYTHON:-python}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs) JOBS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --first) FIRST="$2"; shift 2 ;;
        --last) LAST="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
[[ $# -ge 1 ]] || { echo "usage: $0 [options] -- <runner.py> [runner args]" >&2; exit 2; }

LOG="runs/_logs/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$LOG"
# numpy/pandas would otherwise start one thread per core in every process
export OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
echo "$(date -u +%T) $JOBS jobs x $THREADS threads, years $FIRST-$LAST, seeds $SEEDS: $*"
echo "logs in $LOG"

running=0
for seed in $SEEDS; do
    for year in $(seq "$FIRST" "$LAST"); do
        if (( running >= JOBS )); then wait -n; running=$((running - 1)); fi
        "$PYTHON" -u "$@" --years "$year" --seed "$seed" --threads "$THREADS" \
            > "$LOG/s${seed}_${year}.log" 2>&1 &
        running=$((running + 1))
        echo "$(date -u +%T) started seed $seed year $year"
        sleep 2
    done
done
wait

failed=(); dirs=()
for f in "$LOG"/*.log; do
    if grep -qE '^K= *[0-9]+  ' "$f"; then
        dirs+=($(sed -n 's/^K=[0-9]*: run dir //p' "$f"))
    else
        failed+=("$(basename "$f" .log)")
    fi
done
echo "$(date -u +%T) finished: ${#dirs[@]} run directories, ${#failed[@]} failed ${failed[*]:-}"
printf '%s\n' "${dirs[@]}" > "$LOG/run_dirs.txt"
if (( ${#failed[@]} )); then
    echo "not merging; rerun the failed years (see their logs), then:"
    echo "  $PYTHON scripts/merge_year_runs.py \$(cat $LOG/run_dirs.txt) <reruns>"
    exit 1
fi
"$PYTHON" scripts/merge_year_runs.py "${dirs[@]}"
