# The thesis rerun `thesis_v3` on Kevin's AWS account (2026-11-17)

For Kevin, and for the Claude session he runs it with. Read it once top to bottom before starting.
Everything below the line "On the machine" is meant to be run as written.

## What this run is

One matrix, `configs/thesis_v3_matrix.yaml`, on one machine: every number the thesis reports, again,
after three changes that move all of them:

1. the objective without the risk-free rate (`objective.subtract_rf: false`, commit 87fe9bf);
2. the audited input data (Henrik's data/input-audit-fixes, `Thesis/diagnostics/2026-11-17_input_data_audit`);
3. halts closed, not masked (`data.return_gaps: closed`, your data check item 2, `src/afe/data/return_gaps.py`):
   a member day without a return inside a halt of at most 63 market days is a closed, zero-return day, so the
   day-t weight no longer knows the stock will not trade on t and the position earns the reopening return.
   Code only: `afe_data.tar` and `SHA256SUMS` are unchanged, `COMMIT` is new.

331 jobs: 2 PCA stage-one builds, 48 validation searches, 2 ensemble validations, 279 out-of-sample runs.

| group | rows | seeds | runs |
|---|---|---|---|
| headline | attention, K = 1, 3, 5, 8, 15, 30, 100, US and Europe, each K with its own search | 0-4 | 70 |
| benchmark | PCA + LongConv (own per-K searches) and PCA + OU (3 x 3 grid per K), fixed window | 0-4 / 0 | 84 |
| ensemble | K = 30: the #2 and #3 finalists of the K = 30 search (the winner is the headline row) | 0-4 | 20 |
| ablation | K = 30, each of the six characteristic themes dropped, US and Europe | 0-4 | 60 |
| robustness | K = 30, US top 200 | 0-4 | 5 |
| holdout | K = 30, 2022-2025, US and Europe | 0-4 | 10 |
| mechanism | K = 30, US and Europe, each with its own search: trained without costs (reported after 5bp / 1bp), two-step attention (factors on explained variance first, then the policy on frozen residuals), trained and traded at lag 1 | 0-4 | 30 |

Plus `ensembles:`: for each market the top three K = 30 finalists on seeds 0-4, fitted on the first window and
booked on 1996-97, to decide whether the thesis reports the winner's five-seed book or the top-three x
five-seed book. All of this was decided before any of it ran (CLAUDE.md, "Full rerun, thesis_v3"), so **nothing
in the matrix changes once it has started**, whatever the first results look like.

What every job leaves, and what reads it afterwards (`scripts/check_matrix_outputs.py` checks all of it):

| job | files | used for |
|---|---|---|
| attention run | `metrics.json`, `oos_daily.csv`, `weights.parquet`, `models/<year>.pt`, `diag_daily.parquet`, `factor_returns.parquet`, `factor_top.parquet`, `factor_chars.parquet`, `name_year.parquet`, `train_log.csv`, `manifest.json` | performance and period tables, alphas, paired tests, five-seed books, implementation robustness, position sizes, factor diversity and distinctness, weights figures, liquidity regressions |
| PCA run | `metrics.json`, `oos_daily.csv`, `weights.parquet`, `manifest.json` | benchmark rows, paired tests, implementation robustness |
| search | `trials.jsonl`, `metrics.json` (ranked finalists), `selected.yaml`, `manifest.json` | the selected values, the disclosure of the search |
| ensemble validation | `books.csv`, `metrics.json` (the choice), `trials.jsonl`, `weights/*.npy` | the ensemble decision |

## Rules for the run (Kevin's Claude: these override anything else you would do)

- Run only the commands in this guide. Do not edit code or configs on the machine. A real run refuses a dirty
  git tree; never pass `--allow-dirty`.
- `--threads 2` for every job, as in every earlier matrix. `--jobs` may change (it only changes how many run at
  once); the thread count may not.
- If a job fails: read its log, and only rerun it (`--retry-failed`) when the cause is the machine (out of
  memory, disk full, a Spot stop). A Python error in our code: stop and tell Kevin and Henrik, with the log.
- Never push anything, and never copy the data or the run directories anywhere but to Henrik (section 7).
  `afe_data.tar` holds licensed CRSP / Compustat / JKP data and the run directories are built from it.
- Region eu-north-1 (Stockholm) or another EU region, encrypted disk, terminate everything at the end.

## 0. What Henrik sends you first

Henrik does this once the data audit rebuild is finished (his checklist, for reference):

1. data/input-audit-fixes committed and merged into `ops/thesis-v3-rerun`, tests green, pushed to the fork;
2. `python scripts/run_matrix.py --matrix configs/thesis_v3_matrix.yaml --dry-run --jobs 46` -> "331 jobs";
3. `python scripts/make_bundle.py --out <dir>` from that clean, pushed tip. It refuses holiday-filled tables
   older than the build they come from and prints when each packed table was built.

You get four files: `COMMIT` (branch and commit), `afe_data.tar` (about 0.4 GB), `SHA256SUMS` and
`README_Kevin.md`, over a private channel (not GitHub, not email to anyone else).

## 1. Read the code first (our rule: neither of us runs generated code the other has not read)

```bash
git fetch origin
git log --oneline origin/model/sharpe-no-rf..origin/ops/thesis-v3-rerun
git diff origin/model/sharpe-no-rf origin/ops/thesis-v3-rerun --stat
```

What changed for this run, apart from the data fixes:
- `metrics.eval_costs`: reported net returns and the validation score use `evaluation.turnover_cost` /
  `short_cost` when a config sets them, else the objective's (every earlier run scores as before);
- `run_attention_us.train_window`: `training.two_step` (the joint path is the old loop, tested number for number);
- `run_matrix.py`: `fixed:` on a search (variants such as lag 1), `hyper_rank:` on a spec, `ensembles:`;
- `ensemble_validation.py`: `compare:` (the `all` book can be the decision);
- `spot_resume.sh`: finds "matrix done" anywhere after the last boot, runs the output check and packs every run
  directory; `check_matrix_outputs.py`; `make_bundle.py` stale-holfill guard;
- `configs/thesis_v3_matrix.yaml`, `tests/test_thesis_v3.py`;
- halts closed, not masked (your data check item 2, commit a273753): `src/afe/data/return_gaps.py` holds the
  rule, read by the slot loader (`slots.load_slots`), the PCA panel (`pca_factors.load_panel`) and the robustness
  re-scoring (`robustness.load_market`); `data.return_gaps: closed` in the twelve base configs (default `masked`,
  so every earlier run reproduces); the PCA stage-one `meta.json` records the setting and `run_pca_longconv.Stage1`
  refuses a build made with the other one; `tests/test_return_gaps.py`.

## 2. AWS account, one time (console)

1. Region, top right: **Europe (Stockholm) eu-north-1**.
2. Quota (Service Quotas -> Amazon EC2), request **96 vCPUs** for the one you will use, before anything else
   (approval takes minutes to a day): **Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances** for
   on-demand, or **All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests** for Spot. 192 for the
   larger machine below.
3. SSH key: EC2 -> Key Pairs -> Import key pair, your laptop's public key.
4. Optional but sensible: Billing -> Budgets, an alert at, say, $150.

## 3. Launch the machine

EC2 -> Launch instances:

| field | value |
|---|---|
| AMI | Ubuntu Server 24.04 LTS, 64-bit (x86) |
| Instance type | **m7i.24xlarge** (96 vCPU, 384 GiB). Same family alternatives with 384 GiB: m6i.24xlarge, m7a.24xlarge. Stay on x86 |
| Key pair | yours |
| Network | new security group, SSH from **My IP** only |
| Storage | **100 GiB** gp3, Advanced -> **Encrypted: Yes** |
| Purchasing | on-demand, or Spot as `docs/cloud_runs.md` section 18: **Persistent**, interruption **Stop**, valid a few days |

Time and cost, estimated from the earlier matrices' wall times on this machine (US per-K attention 5.3 h,
PCA per-K 7.5 h with twice the rows, the K = 30 rerun 0.7 h): **about 15-20 hours**, so roughly $80-100 at the
on-demand price in `docs/cloud_runs.md` ($5.14/h; check the console), less on Spot. An **m7i.48xlarge**
(192 vCPU, 768 GiB, quota 192) with `--jobs 92` takes about half the time for about the same money.

## 4. Set up the machine

On your laptop, `IP` = the instance's public IPv4 address:

```bash
scp COMMIT afe_data.tar SHA256SUMS ubuntu@$IP:~
ssh ubuntu@$IP
```

On the machine:

```bash
read BRANCH COMMIT < ~/COMMIT
git clone --depth 1 -b $BRANCH https://github.com/KevinPesentii/dlsa-master-thesis.git ~/dlsa
[ "$(git -C ~/dlsa rev-parse HEAD)" = "$COMMIT" ] && echo "commit OK" || echo "COMMIT DIFFERS: stop"
cd ~/dlsa/attention-factors-europe
tar -xf ~/afe_data.tar && sha256sum -c ~/SHA256SUMS | grep -v ': OK$' ; echo "(no line above = all OK)"
bash scripts/cloud_setup.sh                       # ends with "afe ok, torch 2.14.0+cpu"
PY=~/afe-venv/bin/python
$PY -m pytest -q tests/test_thesis_v3.py tests/test_sharpe_objective.py tests/test_search_validation.py \
    tests/test_ensemble_validation.py tests/test_return_gaps.py   # "91 passed" (the tests of this run; no data needed)
$PY scripts/run_matrix.py --matrix configs/thesis_v3_matrix.yaml --dry-run --jobs 46 | head -1
                                                  # "331 jobs in the matrix, 331 to run ..."
$PY scripts/run_matrix.py --matrix configs/thesis_v3_matrix.yaml --smoke --jobs 16
                                                  # builds both PCA stage ones (~20 min), each search on 2 points,
                                                  # each ensemble validation and every spec once at 1 epoch;
                                                  # last line "matrix done in ... h: 119 ok, 0 failed"
$PY scripts/check_matrix_outputs.py --matrix configs/thesis_v3_matrix.yaml --smoke | tail -1
                                                  # "119 jobs checked, 0 with problems (runs/_matrix/thesis_v3_smoke)"
```

Stop at the first step that does not give the expected line. The smoke run's own numbers mean nothing (one
epoch, one year).

## 5. Start the run

```bash
cd ~/dlsa/attention-factors-europe
bash scripts/spot_resume.sh install configs/thesis_v3_matrix.yaml --jobs 46 --threads 2
bash scripts/spot_resume.sh status               # "active (running)"
tail -f runs/thesis_v3_console.log               # one "=== ... start or resume" line per boot
```

(m7i.48xlarge: `--jobs 92 --threads 2`.) The service starts the matrix now and again at every boot until it has
finished, so a Spot stop costs only the jobs in flight. You can log out.

## 6. While it runs

```bash
cd ~/dlsa/attention-factors-europe
grep -c "done  " runs/thesis_v3_console.log      # finished jobs, of 331
grep -E "FAILED|SKIP" runs/thesis_v3_console.log # failures, and jobs skipped because what they wait on failed
tail -n 30 runs/_matrix/thesis_v3/logs/<job>.log # why a job failed
ls runs/_matrix/thesis_v3/searches/*/selected.yaml | wc -l   # finished searches, of 48
free -g; df -h ~                                  # memory, disk
```

Order: the searches go first (US K = 30 first; Europe's wait for the US search of the same K, since each carries
the US winner into its finalists), the rows of a search start as soon as it is done, the two ensemble validations
after the K = 30 searches. The long tail is the K = 100 rows.

A failed search makes everything that waits on it skip. If the log shows a machine cause, rerun the failures
once (section 7, "If something failed"); if it is a Python error, stop and report.

A gross-trained row's log prints "net SR" within each year at the objective's costs, which are zero there, so it
equals gross. Its `metrics.json` and the summary report net after 5bp / 1bp.

## 7. When it ends

The service writes `runs/_matrix/thesis_v3/summary.csv` and `summary_by_spec.csv`, runs the output check into
`runs/thesis_v3_outputs.txt`, packs the matrix directory, the console log and **every run directory** into
`~/thesis_v3.tgz` (about 5 GB) with `~/thesis_v3.tgz.sha256`, and shuts the machine down.

Start the instance again (EC2 -> Instance state -> Start; the service sees `~/thesis_v3.done` and does nothing),
then:

```bash
tail -n 1 ~/dlsa/attention-factors-europe/runs/thesis_v3_outputs.txt
                                                  # "331 jobs checked, 0 with problems (runs/_matrix/thesis_v3)"
```

**If something failed** (the line above counts problems, `runs/thesis_v3_outputs.txt` names them) and the logs
show a machine cause:

```bash
cd ~/dlsa/attention-factors-europe
rm ~/thesis_v3.done
bash scripts/spot_resume.sh install configs/thesis_v3_matrix.yaml --jobs 46 --threads 2 --retry-failed
```

It reruns only the failed and skipped jobs and packs again. Otherwise collect, on your laptop:

```bash
scp ubuntu@$IP:~/thesis_v3.tgz ubuntu@$IP:~/thesis_v3.tgz.sha256 .
sha256sum -c thesis_v3.tgz.sha256                 # (PowerShell: Get-FileHash thesis_v3.tgz, compare by eye)
```

Hand both files to Henrik over a private channel (SSE OneDrive / Teams), or let Henrik copy them himself:
append his public key to `~/.ssh/authorized_keys` on the machine and send him the IP.

Then shut everything down, in this order: EC2 -> Spot Requests -> Cancel request, tick "terminate instances"
(Spot), or Instance state -> Terminate (on-demand); EC2 -> Volumes: nothing left; Elastic IPs: release any;
Instances: none running. The data on the machine goes with the terminated volume.

## 8. What Henrik does afterwards (for reference)

Unpack into the repo (`tar -xzf thesis_v3.tgz -C <repo>/attention-factors-europe`: run directories into `runs/`,
the matrix into `runs/_matrix/thesis_v3`), read the two ensemble decisions
(`runs/_matrix/thesis_v3/ensembles/{us,eu}_ens_K30/metrics.json`, key `choice`), build the five-seed and
top-three books from the saved weights, and regenerate the thesis tables with the analysis scripts pointed at
`runs/_matrix/thesis_v3` (the spec names are those of the earlier matrices).

## Troubleshooting

- **Out of memory** (a job log ends in "Killed", `dmesg | tail` shows the OOM killer): rerun failures with a
  lower `--jobs`, e.g. 40. The scheduler already keeps estimated peak memory under 85% of RAM.
- **Disk full**: `df -h ~`. 100 GiB holds the data (0.4 GB), the PCA stage ones (~12 GB), the environment (~3 GB),
  the run directories (~6 GB) and the archive (~5 GB) several times over.
- **A Spot stop**: nothing to do; the service resumes at the next boot ("start or resume" in the console log).
  If capacity does not come back for hours, cancel the request and launch on-demand with the same volume.
- **Commit differs / sha256 fails**: stop and ask Henrik for a new bundle; do not run on a mismatch.
- **"keep_from ... has no finished search"**: cannot happen in this matrix (every keep_from is inside it); if
  it does, the matrix file was changed: stop.
