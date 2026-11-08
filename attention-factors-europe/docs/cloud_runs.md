# Running on a rented machine (AWS spot)

A full 1998-2021 run trains 24 independent annual models one after another: about an hour
on the laptop at K=30, and a multiple of that for several seeds or yearly lambda
selection. The years do not depend on each other, so on a machine with many cores every
year can be its own process and the whole run takes about as long as its slowest year.

- `scripts/run_years_parallel.sh` starts one process per year (and seed), N at a time,
  and merges the results.
- `scripts/merge_year_runs.py` puts the per-year run directories back into one. The only
  number that changes is turnover on each year's first date: a standalone year starts from
  no position, the sequential runner from last year's book. Merged metrics equal the
  sequential ones (checked on the K=1 run of 2026-09-24: largest daily turnover difference
  1.4e-7, float32 rounding; and end to end, validation runner on 2020-2021 at one epoch:
  parallel and sequential print the same metrics line).
- `scripts/cloud_setup.sh` installs the pinned environment on Ubuntu 24.04.

Why CPU and not GPU: the runners never move tensors to a device, and one training step is
small (500 names, 30 factors, 125 days), so a GPU would need code changes and would still
lose to 24 CPU processes.

Which runners reproduce a sequential run: `run_attention_us_val.py` seeds every year from
(seed, year), so parallel and sequential give the same numbers. `run_attention_us.py`
seeds once per process, so every standalone year starts from the same draw; that is a
valid run but not the same numbers as its sequential version.

## 0. Before the first run: licence

`returns.parquet` holds CRSP returns and `features.parquet` is built from CRSP and
Compustat. Check with the SSE library that the WRDS subscription allows processing on a
cloud machine you control. Whatever the answer, keep to an EU region, encrypt the disk,
never share the machine, and terminate it (which deletes the disk) when done.

## 1. One-time AWS setup (console, about 15 minutes plus the quota wait)

1. Sign in at console.aws.amazon.com. Region, top right: **Europe (Stockholm) eu-north-1**.
2. **Spot quota.** New accounts may only run a few spot vCPUs. Service Quotas -> AWS
   services -> Amazon Elastic Compute Cloud (Amazon EC2) -> **All Standard (A, C, D, H, I,
   M, R, T, Z) Spot Instance Requests** -> Request increase at account level -> 96.
   Approval takes minutes to a day, so do this first.
3. **SSH key.** Use the laptop key already in the Windows agent. In PowerShell:
   `Get-Content $HOME\.ssh\id_ed25519.pub | Set-Clipboard`. Then EC2 -> Key Pairs ->
   Actions -> Import key pair, name `henrik-laptop`, paste, Import. The private key never
   leaves the laptop.

## 2. Launch the machine (every run, about 5 minutes)

EC2 -> Instances -> **Launch instances**:

| field | value |
|---|---|
| Name | `afe-run` |
| AMI | Ubuntu Server 24.04 LTS, 64-bit (x86) |
| Instance type | `c7i.24xlarge` (96 vCPU, 192 GiB). If not offered or pricey: `c6i.24xlarge`, `c6a.24xlarge`, `m7i.24xlarge`. Stay on x86 (Intel/AMD), like the laptop |
| Key pair | `henrik-laptop` |
| Network settings | Create security group, Allow SSH traffic from **My IP** only |
| Storage | 30 GiB gp3; Advanced -> Encrypted: Yes |
| Advanced details -> Purchasing option | tick **Spot instances**, keep the defaults (one-time, terminate on interruption) |

Launch, open the instance, copy its **Public IPv4 address**. Current prices: EC2 -> Spot
Requests -> Pricing history. Sizing rule: jobs = min(vCPUs / 4, RAM GiB / 5). Each
process of the validation runner loads 1990 to its year and peaked at 4.4 GB for 2021.

## 3. Send code and data (PowerShell on the laptop)

The code goes as a git bundle, so run directories on the machine still record the commit.
Bundle the branch you want to run; it must be committed.

```powershell
$IP = "1.2.3.4"   # the Public IPv4 address
cd C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-integration
git bundle create $env:TEMP\afe.bundle ops/parallel-year-runs
tar -cf $env:TEMP\us_shared.tar -C C:\Users\henri\Desktop\SSE\masterThesis\attention-factors-europe\data\us\shared returns.parquet universe.parquet features.parquet
scp $env:TEMP\afe.bundle $env:TEMP\us_shared.tar "ubuntu@${IP}:~"
ssh "ubuntu@$IP"
```

The first `scp` asks to trust the host key: type `yes`. The data is about 126 MB.

## 4. Set up the machine (on the VM, about 5 minutes)

```bash
git clone -b ops/parallel-year-runs ~/afe.bundle ~/dlsa      # sudo apt-get install -y git if missing
cd ~/dlsa/attention-factors-europe
mkdir -p data/us/shared && tar -xf ~/us_shared.tar -C data/us/shared
bash scripts/cloud_setup.sh                                  # ends with "afe ok, torch 2.14.0+cpu"
export PYTHON=~/afe-venv/bin/python
nproc && free -g
```

Smoke test, two years at one epoch (numbers meaningless, a few minutes):

```bash
bash scripts/run_years_parallel.sh --jobs 2 --threads 4 --first 2020 --last 2021 -- \
    scripts/run_attention_us_val.py --K 30 --epochs 1 --grid 0.1
```

It should end with `finished: 2 run directories, 0 failed` and a merged metrics line.

## 5. The run (on the VM)

Run inside tmux so a dropped SSH connection does not kill it:

```bash
tmux new -s afe
export PYTHON=~/afe-venv/bin/python
bash scripts/run_years_parallel.sh --jobs 24 --threads 4 -- \
    scripts/run_attention_us_val.py --K 30 --select once
```

Detach with `Ctrl-b` then `d`; come back with `tmux attach -t afe`. Watch with `htop`, or
`tail -f runs/_logs/*/s0_2021.log`. Every process first repeats the lambda selection on
1990-1997 (the same computation in each, so it costs money, not time), then trains its
year. Expect roughly 4 laptop refits of wall time, 10-20 minutes; not yet measured on AWS.

Variants: more seeds with `--seeds "0 1 2 3 4"` (Table 3 of the paper averages seeds; 120
processes, 24 at a time); the score target jointly with `--tau-grid 0.5 1 2`; a fixed
lambda with `-- scripts/run_attention_us.py --K 30 --config <config>`.

The last lines print the merged metrics and the merged run directory,
`runs/<stamp>_attention_val_K30_s0_merged/`. That directory is the result to cite; its
manifest lists the per-year directories it came from.

## 6. Bring the results home and shut down

On the VM: `tar -czf ~/runs.tgz -C ~/dlsa/attention-factors-europe runs`. Then in
PowerShell on the laptop:

```powershell
scp "ubuntu@${IP}:~/runs.tgz" $env:TEMP\
tar -xzf $env:TEMP\runs.tgz -C C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-integration\attention-factors-europe
```

Then EC2 -> Instances -> select -> Instance state -> **Terminate**. Check that EC2 ->
Volumes is empty and EC2 -> Spot Requests has nothing open. Until the instance is
terminated it is billed, whether or not anything runs.

## If AWS takes the machine back

Spot instances can be reclaimed with two minutes' notice, and everything on the disk
goes with them, finished years included. Runs are short, so relaunch and start over.
(For a long matrix see section 18: a persistent request that stops instead of terminating.) If
it keeps happening, pick another instance type or availability zone, or tick nothing
under Purchasing option (on-demand) for the 20 minutes. A single year that failed can be
rerun alone (`--first Y --last Y`) and merged with the others:
`$PYTHON scripts/merge_year_runs.py $(cat runs/_logs/<stamp>/run_dirs.txt) runs/<rerun dir>`.

Numbers from different CPUs differ in the last digits (float rounding), which is far
below seed noise.

## 7. The thesis matrix (2026-10-10)

`configs/thesis_matrix.yaml` is the whole set of runs behind the thesis tables, fixed
before any of it ran: the headline grid (attention, PCA + LongConv, PCA + OU, every K, five
seeds, both markets), the characteristic-group ablation, the robustness rows and the
2022-2025 holdout: 239 runs. `scripts/run_matrix.py` runs them on one machine: each job is
one full sequential runner process (all years, so lagged and held positions cross year
ends exactly as on the laptop; no year merging), longest first, as many at a time as
`--jobs` and the memory budget allow. The two PCA stage ones are built on the machine first.

Every return uses validated hyperparameters (fit 1990-1995, net Sharpe on 1996-1997). The
attention headline values are already in `configs/us_headline.yaml` / `europe_headline.yaml`
(the US search, run on the laptop). Five more searches run on the machine before the runs
that need them: PCA + LongConv, PCA + OU thresholds, attention with raw input, attention at
lag 1, PCA + LongConv at lag 1 (`searches:` in the matrix; each uses 8-12 job slots for
roughly 45 minutes). The runs that do not wait on a search (about 140, the long US
attention grid among them) start at once. `runs/_matrix/thesis/searches.json` lists what
each search selected.

Size: about 150 laptop process-hours by the matrix's own rough cost model (US attention
K=30 about an hour per job at 2 threads; Europe about half; K=100 about twice). Memory:
about 5.5 GB per US attention job, 3 GB per Europe job (estimates with margin; the US
validation runner peaked at 4.4 GB). Machine: **m7i.24xlarge** (96 vCPU, 384 GiB) runs 46
jobs x 2 threads without hitting the memory budget, so expect roughly 3-5 hours. A
c7i.24xlarge (192 GiB) also works; the budget then keeps about 28 US jobs in flight.
Storage: **60 GiB** gp3, encrypted (the two PCA stage ones are about 12 GB, the 239 run
directories about 3 GB, the environment about 3 GB).
Because a reclaimed spot machine loses everything, prefer **on-demand** for this run, or
copy `runs/` home every hour or two (the ledger lets a new machine skip what came back).

On the laptop (PowerShell), from the branch to run, committed and pushed:

```powershell
cd C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-europe\attention-factors-europe
$PY = "C:\Users\henri\miniconda3\envs\afe\python.exe"
& $PY scripts\run_matrix.py --dry-run | Select-Object -First 3    # 246 jobs: 2 PCA builds, 5 searches, 239 runs
& $PY scripts\make_bundle.py --out $env:TEMP\afe_bundle             # ~0.3 GB: afe_data.tar, SHA256SUMS, COMMIT
$IP = "1.2.3.4"
scp $env:TEMP\afe_bundle\afe_data.tar $env:TEMP\afe_bundle\SHA256SUMS $env:TEMP\afe_bundle\COMMIT "ubuntu@${IP}:~"
ssh "ubuntu@$IP"
```

The code is cloned on the machine from the public fork; only the licensed data is copied.
(Without GitHub access: `make_bundle.py --git-bundle`, copy `afe.bundle` too, and clone
from `~/afe.bundle` instead.)

On the machine:

```bash
read BRANCH COMMIT < ~/COMMIT
git clone --depth 1 -b $BRANCH https://github.com/KevinPesentii/dlsa-master-thesis.git ~/dlsa
[ "$(git -C ~/dlsa rev-parse HEAD)" = "$COMMIT" ] && echo "commit OK" || echo "COMMIT DIFFERS: stop"
cd ~/dlsa/attention-factors-europe
tar -xf ~/afe_data.tar && sha256sum -c ~/SHA256SUMS     # every line "OK"
bash scripts/cloud_setup.sh                              # ends with "afe ok, torch 2.14.0+cpu"
PY=~/afe-venv/bin/python
$PY scripts/run_matrix.py --smoke --jobs 16              # builds both PCA stage ones (~20 min), each search
                                                         # on 2 points, every spec once at 1 epoch; "0 failed"
tmux new -s afe
$PY scripts/run_matrix.py --jobs 46 --threads 2 2>&1 | tee runs/matrix_console.log
```

Progress: `tail -f runs/matrix_console.log`, `htop`, one log per job in
`runs/_matrix/thesis/logs/`. Restarting the same command skips finished jobs; add
`--retry-failed` to rerun failures (their logs say why). `--groups headline` (or ablation,
robustness, extension) runs a subset. At the end it prints a table by spec and writes
`runs/_matrix/thesis/summary.csv` (one row per run) and `summary_by_spec.csv` (mean and sd
over seeds).

Bring everything home (run directories hold the daily series, weights and yearly models;
several GB) and terminate the machine:

```bash
tar -czf ~/runs_thesis.tgz -C ~/dlsa/attention-factors-europe runs
```
```powershell
scp "ubuntu@${IP}:~/runs_thesis.tgz" $env:TEMP\
tar -xzf $env:TEMP\runs_thesis.tgz -C C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-europe\attention-factors-europe
```

## 8. The Europe-validated matrix (2026-10-12)

`configs/thesis_matrix_europe.yaml` reruns the 121 Europe runs of section 7 with
hyperparameters validated on EUROPEAN data. Its six searches are the US ones
(`configs/us_search*.yaml`) with only the data changed (`configs/europe_search_*.yaml`):
every candidate is fitted on 1993-1995 and scored by its net Sharpe on 1996-1997, the same
Sobol points in the same order, and each search carries its US winner into the finalists.
The runs wait for their search; `runs/_matrix/thesis_eu/searches.json` lists the winners.
Spec names are section 7's, so the Europe rows with the US values (`runs/_matrix/thesis`)
stay the comparison.

Size: in the section 7 run the Europe jobs took 4,242 process-minutes, the longest 92
(attention K = 100); the six searches together fit in the 46 slots and take roughly 15-40
minutes. Expect about 2-2.5 hours on 46 slots, more if a winner trains slower (e.g. batch
63). Memory is about 3 GB per job, so a **c7i.24xlarge** (192 GiB) is enough; the
m7i.24xlarge works as before. Storage: **40 GiB** (the Europe PCA stage one is 5.7 GB).

Everything else is section 7 with three changes: the bundle carries only the European
tables (`--markets europe`, about 0.15 GB, no CRSP data), every `run_matrix.py` command
names the matrix, and the results come home under their own name.

```powershell
& $PY scripts\run_matrix.py --matrix configs\thesis_matrix_europe.yaml --dry-run | Select-Object -First 1   # 128 jobs: 1 PCA build, 6 searches, 121 runs
& $PY scripts\make_bundle.py --markets europe --out $env:TEMP\afe_bundle_eu
scp $env:TEMP\afe_bundle_eu\afe_data.tar $env:TEMP\afe_bundle_eu\SHA256SUMS $env:TEMP\afe_bundle_eu\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7:

```bash
$PY scripts/run_matrix.py --matrix configs/thesis_matrix_europe.yaml --smoke --jobs 16   # PCA stage one (~5 min), "0 failed"
tmux new -s afe
$PY scripts/run_matrix.py --matrix configs/thesis_matrix_europe.yaml --jobs 46 --threads 2 2>&1 | tee runs/matrix_eu_console.log
tar -czf ~/runs_thesis_eu.tgz -C ~/dlsa/attention-factors-europe runs
```

At home, unpack `runs_thesis_eu.tgz` as in section 7: the run directories have new
timestamped names and the ledger is `runs/_matrix/thesis_eu`, so nothing of the first
matrix is overwritten.

## 9. The paper-gap matrix (2026-10-16)

`configs/gap_matrix.yaml` tests the explanations of the gap to the paper in
`Thesis/diagnostics/2026-10-16_paper_gap`. Each row is the US headline (sobol-036) with one
change:
- the 800 largest listings instead of 500 (`configs/us_data_top800.yaml`, built on the laptop
  into `data/us/top800`);
- λ_ridge 1e-3;
- no score temperature;
- a paper-literal setting.

The headline itself runs too, as the paired baseline. Every row runs at K = 8 / 30 / 100 and
seeds 0-2 with the yearly models saved: 45 runs, no prep jobs, no searches. Diagnostic rows
only: nothing is selected on them.

Size, from the section 7 run (2 threads, 46 jobs in flight): US attention jobs took about
90 / 115 / 205 minutes at K = 8 / 30 / 100.
- The 800-name jobs do about 1.6 times the work (`cost_scale` in the matrix). The longest,
  K = 100 on 800 names, needs roughly 5-6 hours and sets the wall-clock. Everything else is
  done within about 3.5 hours.
- Memory: 5.5 GB per 500-name job and 10 GB per 800-name job (`mem_gb` in the matrix; the
  full 800-name panel load peaked at 7.9 GB on the laptop), about 290 GB for all 45. Use the
  **m7i.24xlarge** (384 GiB) again: all 45 jobs fit its budget at once. On a c7i.24xlarge
  (192 GiB) the budget holds about half of them back and the run takes longer.
- Storage: 30 GiB (the bundle is about 0.4 GB, the 45 run directories about 2 GB).

```powershell
& $PY scripts\run_matrix.py --matrix configs\gap_matrix.yaml --dry-run | Select-Object -First 1   # 45 jobs
& $PY scripts\make_bundle.py --markets us us_top800 --out $env:TEMP\afe_bundle_gap
scp $env:TEMP\afe_bundle_gap\afe_data.tar $env:TEMP\afe_bundle_gap\SHA256SUMS $env:TEMP\afe_bundle_gap\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7:

```bash
$PY scripts/run_matrix.py --matrix configs/gap_matrix.yaml --smoke --jobs 8   # 5 rows once, 1 epoch, 1998: "0 failed"
tmux new -s afe
$PY scripts/run_matrix.py --matrix configs/gap_matrix.yaml --jobs 46 --threads 2 2>&1 | tee runs/matrix_gap_console.log
tar -czf ~/runs_gap.tgz -C ~/dlsa/attention-factors-europe runs
```

At home, unpack `runs_gap.tgz` as in section 7. The results are under `runs/_matrix/gap`
(ledger, `summary.csv`, `summary_by_spec.csv`), and the run directories have new
timestamped names, so nothing earlier is overwritten.

## 10. The second paper-gap matrix (2026-10-19)

`configs/gap2_matrix.yaml` asks why the 800 largest listings helped in the run of section 9
(`runs/_matrix/gap`: K = 30 net Sharpe 1.50 -> 1.84, no gain at K = 8). The 800 names are our 500
plus 300 smaller firms, so that run could not tell breadth (more names per factor) from size (the
smaller firms' returns). Four rows, each at K = 8 / 30 / 100 and seeds 0-2 with the yearly models
saved: 36 runs, no prep jobs, no searches. Diagnostic rows only: nothing is selected on them.
- `us_headline` and `us_top800`: the rows of section 9 again, as baselines from the same machine;
- `us_rank301_800`: the headline on the 500 listings ranked 301-800
  (`configs/us_data_rank301_800.yaml`, built on the laptop into `data/us/rank301_800`; 2021 caps
  $9.5-42bn, against $18.5bn and up for the headline; ranks 301-500 are in both). The headline's
  breadth with the 800-name build's smaller firms;
- `us_top800_paper_literal`: section 9's paper-literal setting on the 800 names. That setting had
  the paper's σ and cost drag but lost gross Sharpe; the 800 names gained gross Sharpe.

Machine: **m7i.8xlarge** on-demand (32 vCPU, 128 GiB), as for the run of 2026-10-18, with
`--jobs 16 --threads 2`.
- Durations on that machine (2026-10-18 ledger): 500-name jobs about 49 / 61 / 115 minutes at
  K = 8 / 30 / 100, 800-name jobs about 57 / 74 / 206.
- Memory: half the jobs here have 800 names (10 GB each against 5.5), so the budget (85% of RAM,
  about 105 GB) keeps 11-14 jobs in flight rather than 16.
- A simulation of the scheduler with those durations gives about 56 process-hours and **about
  5 hours** wall-clock, roughly $9 at the Stockholm on-demand price ($1.71/h). The six 800-name
  K = 100 jobs (about 3.5 hours each) start first and set the floor.
- Storage: 30 GiB (bundle about 0.5 GB, the 36 run directories about 2 GB).

On the laptop:

```powershell
& $PY scripts\run_matrix.py --matrix configs\gap2_matrix.yaml --dry-run | Select-Object -First 1   # 36 jobs
& $PY scripts\make_bundle.py --markets us us_top800 us_rank301_800 --out $env:TEMP\afe_bundle_gap2
scp $env:TEMP\afe_bundle_gap2\afe_data.tar $env:TEMP\afe_bundle_gap2\SHA256SUMS $env:TEMP\afe_bundle_gap2\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7:

```bash
$PY scripts/run_matrix.py --matrix configs/gap2_matrix.yaml --smoke --jobs 8   # 4 rows once, 1 epoch, 1998: "0 failed"
tmux new -s afe
$PY scripts/run_matrix.py --matrix configs/gap2_matrix.yaml --jobs 16 --threads 2 2>&1 | tee runs/matrix_gap2_console.log; tar -czf ~/runs_gap2.tgz -C ~/dlsa/attention-factors-europe runs; sudo shutdown -h now
```

The last line archives `runs/` and stops the machine when the matrix ends, so an unattended run
costs only its own hours (a stopped instance is billed for its volume only, a few cents a day). To
collect: EC2 -> Instance state -> **Start** (it gets a new public IP), then in PowerShell

```powershell
scp "ubuntu@${IP}:~/runs_gap2.tgz" $env:TEMP\
tar -xzf $env:TEMP\runs_gap2.tgz -C C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-europe\attention-factors-europe
```

and **Terminate** the instance (check that EC2 -> Volumes is empty). The results are under
`runs/_matrix/gap2` (ledger, `summary.csv`, `summary_by_spec.csv`); the run directories have new
timestamped names, so nothing earlier is overwritten. The leg replay of the saved seed-0 models
runs on the laptop afterwards (`replay_legs.py ... --matrix gap2 --spec <row>` in the diagnostics
folder; it loads one year of the panel at a time, so 800 names fit in memory).

## 11. The fixed-window experiment (2026-11-01)

`configs/exp_fixedwin_matrix.yaml` asks whether the policy should see the past of the residual
portfolio it actually trades. The headline (`policy.residual_window: rolling`) hedges each day of
the 30-day window with that day's factor weights and loadings; `us_fixedwin`
(`configs/us_headline_fixedwin.yaml`) re-hedges the whole window with the trade date's
(`src/afe/model/attention_pipeline.py`, `fixed_residual_windows`). Two rows, the US headline and the
headline with only that change, each at K = 8 / 30 / 100 and seeds 0-1: 12 runs, no prep jobs, no
searches. Outside the thesis rows: nothing is selected on them.

Machine: **m7i.8xlarge** on-demand (32 vCPU, 128 GiB), as in section 10, with `--jobs 12 --threads 2`:
every job starts at once.
- Durations on that machine (2026-10-18 ledger): headline jobs about 49 / 61 / 115 minutes at
  K = 8 / 30 / 100. The fixed window adds
  no measurable time: on the laptop (1998, K = 30, the two side by side at 3 threads) one refit took
  215 s against 213 s for rolling. Not measured at K = 8 / 100; the extra work is K x 30 factor returns
  per date.
- Memory: 5.5 GB per job, 66 GB for all 12, inside the budget (85% of RAM).
- Wall-clock is set by the four K = 100 jobs: about 2 hours, so roughly 2.5 hours with setup and
  about $4 at the Stockholm on-demand price ($1.71/h).
- Storage: 30 GiB (bundle about 0.3 GB, the 12 run directories well under 1 GB).

**Faster: one job per year (`--split-years`).** Every run is 24 independent annual refits done one
after the other, which is what makes a K = 100 job take 2 hours. With `--split-years` each year is
its own job (288 in all, about 5 minutes each at K = 100 on that machine), and a merge job per run
(`scripts/merge_year_runs.py`, seconds) puts the years back together under the run's id, with the
turnover at each year seam charged from the previous year's book. The summary is as before.
- The numbers are NOT those of a sequential run: `run_attention_us.py` seeds once per process, so
  every year starts from the same draw. Both rows run split, so the paired comparison holds; do not
  set split rows beside the sequential ones of `runs/_matrix/gap` / `gap2`.
- Work: about 900 process-minutes of training (the sequential durations above, 2 x 2 x (49 + 61 +
  115)) plus 288 panel loads of about 20 s.
- Memory: a year job peaked at 2.0 GB (2.4 GB private) on the laptop (K = 100, 2021); the scheduler
  counts 3 GB.
- **m7i.24xlarge** on-demand (96 vCPU, 384 GiB, $5.14/h), `--jobs 48 --threads 2`: an estimated
  25-40 minutes for the matrix (the range allows for a crowded machine running each job slower, as
  the 46-job run of section 7 did), well under an hour with setup, about $5. Needs an on-demand quota
  of 96 vCPUs (Service Quotas -> "Running On-Demand Standard instances").
- With the 32-vCPU quota: the **m7i.8xlarge** with `--split-years --jobs 16 --threads 2`, an estimated
  1-1.25 hours instead of about 2.

On the laptop, from the branch, committed and pushed:

```powershell
& $PY scripts\run_matrix.py --matrix configs\exp_fixedwin_matrix.yaml --dry-run | Select-Object -First 1   # 12 jobs
& $PY scripts\make_bundle.py --markets us --out $env:TEMP\afe_bundle_fixedwin
scp $env:TEMP\afe_bundle_fixedwin\afe_data.tar $env:TEMP\afe_bundle_fixedwin\SHA256SUMS $env:TEMP\afe_bundle_fixedwin\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7:

```bash
$PY scripts/run_matrix.py --matrix configs/exp_fixedwin_matrix.yaml --smoke --jobs 2   # 2 rows once, 1 epoch, 1998: "0 failed"
tmux new -s afe
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python   # again inside tmux: PY does not carry over
$PY scripts/run_matrix.py --matrix configs/exp_fixedwin_matrix.yaml --jobs 12 --threads 2 2>&1 | tee runs/matrix_fixedwin_console.log; tar -czf ~/runs_fixedwin.tgz -C ~/dlsa/attention-factors-europe runs; sudo shutdown -h now
```

Split into years instead (m7i.24xlarge; on the m7i.8xlarge `--jobs 16`):

```bash
$PY scripts/run_matrix.py --matrix configs/exp_fixedwin_matrix.yaml --smoke --split-years --jobs 4   # 1998 + 1999 per row, merged: "0 failed"
tmux new -s afe
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python   # again inside tmux: PY does not carry over
$PY scripts/run_matrix.py --matrix configs/exp_fixedwin_matrix.yaml --split-years --jobs 48 --threads 2 2>&1 | tee runs/matrix_fixedwin_console.log; tar -czf ~/runs_fixedwin.tgz -C ~/dlsa/attention-factors-europe runs; sudo shutdown -h now
```

Run one or the other into a matrix directory, never both: a restart counts a run as done when its
id has a finished entry in the ledger, whichever way it ran. The last line archives `runs/` and
stops the machine when the matrix ends. To collect: EC2 ->
Instance state -> **Start** (new public IP), then in PowerShell

```powershell
scp "ubuntu@${IP}:~/runs_fixedwin.tgz" $env:TEMP\
tar -xzf $env:TEMP\runs_fixedwin.tgz -C C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-europe\attention-factors-europe
```

and **Terminate** the instance (check that EC2 -> Volumes is empty). The results are under
`runs/_matrix/exp_fixedwin` (ledger, `summary.csv`, `summary_by_spec.csv`); compare `us_fixedwin`
with `us_headline` seed by seed.

## 12. The fixed-window experiment, full sequential run (2026-11-01)

`configs/exp_fixedwin_full_matrix.yaml` repeats the thesis headline row as it ran (`us_attention` of
`configs/thesis_matrix.yaml`: K = 1 / 3 / 5 / 8 / 15 / 30 / 100, seeds 0-4, every year in one process,
yearly models saved; net Sharpe 1.594 at K = 30 in `runs/_matrix/thesis`) twice:
- `us_attention`: the rolling window with the headline values (sobol-036), unchanged. It is there as
  a baseline from the same machine: the same config gave K = 30 net 1.59 on the m7i.24xlarge of
  2026-10-10 and 1.50 on the m7i.8xlarge of 2026-10-18, half the effect being tested;
- `us_attention_fixedwin`: `policy.residual_window: fixed` with its own validated values. Like the
  raw-input and lag-1 rows of the thesis, it gets the headline's search with only that change held
  fixed: `configs/us_search_fixedwin.yaml` (= `configs/us_search.yaml` plus `fixed:
  policy.residual_window: fixed`; 128 Sobol points, fit 1990-1995, net Sharpe on 1996-1997, best mean
  of the top 5 over seeds 0-2). The search runs on the machine first (12 workers); the fixed row
  waits for it and runs with its winner (`hyper: us_fixedwin`); the baseline starts at once.

71 jobs: 1 search and 70 runs. Diagnostic rows, outside the thesis: nothing is selected on the
out-of-sample years. `--groups fixedwin` runs the search and the fixed row only (35 runs, compared
with `runs/_matrix/thesis` across machines).

Size, from the ledgers of the thesis run (m7i.24xlarge, 46 jobs: US attention K = 1-100 about 85-208
minutes, the US attention searches about 40 minutes) and of 2026-10-18 (m7i.8xlarge, 16 jobs: K = 8 /
30 / 100 about 49 / 61 / 115). A simulation of the scheduler with those durations:
- **m7i.8xlarge** on-demand (32 vCPU, 128 GiB), `--jobs 16 --threads 2`: about **5.5 hours**, about
  $10. The search holds 12 of the 16 slots for its first hour, then the 66 remaining runs share them.
- **m7i.24xlarge** on-demand (96 vCPU, 384 GiB), `--jobs 46 --threads 2`, the type of the thesis run:
  about **4.2 hours**, about $23; needs an on-demand quota of 96 vCPUs.
- Memory: the search 30 GB (12 x 2.5), runs 5.5 GB each; inside the budget on both.
- Storage: 30 GiB (the 70 run directories with their yearly models about 2-3 GB).

On the laptop, from the branch, committed and pushed:

```powershell
& $PY scripts\run_matrix.py --matrix configs\exp_fixedwin_full_matrix.yaml --dry-run | Select-Object -First 1   # 71 jobs
& $PY scripts\make_bundle.py --markets us --out $env:TEMP\afe_bundle_fixedwin_full
scp $env:TEMP\afe_bundle_fixedwin_full\afe_data.tar $env:TEMP\afe_bundle_fixedwin_full\SHA256SUMS $env:TEMP\afe_bundle_fixedwin_full\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7:

```bash
~/afe-venv/bin/python scripts/run_matrix.py --matrix configs/exp_fixedwin_full_matrix.yaml --smoke --jobs 4   # search on 2 points, each row once: "0 failed"
tmux new -s afe
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python   # again inside tmux: PY does not carry over
$PY scripts/run_matrix.py --matrix configs/exp_fixedwin_full_matrix.yaml --jobs 16 --threads 2 2>&1 | tee runs/matrix_fixedwin_full_console.log; tar -czf ~/runs_fixedwin_full.tgz -C ~/dlsa/attention-factors-europe runs; sudo shutdown -h now
```

(`--jobs 46` on the m7i.24xlarge.) Collect as in section 11 (`runs_fixedwin_full.tgz`). The results are
under `runs/_matrix/exp_fixedwin_full` (ledger, `summary.csv`, `summary_by_spec.csv`, `searches.json`
with the fixed row's selected values, `searches/us_fixedwin/` with every trial); compare
`us_attention_fixedwin` with `us_attention` seed by seed.

## 13. Seed ensemble against config ensemble, K = 30 (2026-11-02)

One book holding the mean of the five seeds' weights beat every single seed out of sample
(`runs/_matrix/exp_fixedwin_full/seed_ensemble.py`: K = 30 net 1.58 -> 1.97 rolling, 1.67 -> 2.06 fixed).
Before that idea goes anywhere, the validation window has to choose it: does the protocol prefer the
search winner on five seeds, or the search's top five configurations side by side?

1. `scripts/ensemble_validation.py` on the first training window (fit 1990-1995, booked on 1996-1997),
   once per residual window (`configs/us_ensemble_val.yaml`, `configs/us_ensemble_val_fixedwin.yaml`):
   the five best finalists of that window's search x seeds 0-4 = 25 search trials each, whose
   validation books are averaged both ways. The rule with the higher validation net Sharpe wins
   (`metrics.json` -> `choice`); `books.csv` has every book.
2. `configs/exp_ensemble_k30_matrix.yaml`: the out-of-sample runs either answer needs, the four other
   candidates of each window at K = 30, seeds 0-4, 1998-2021 (40 runs). The winner's runs are in
   `runs/_matrix/exp_fixedwin_full`; the books are put together at home.

Size, from the K = 30 runs of the two last matrices (63 minutes each on the m7i.8xlarge at 16 jobs,
119 on the m7i.24xlarge at 46) and one search trial about 1.5 minutes:
- **m7i.8xlarge** on-demand, `--jobs 16`: validation about 10 minutes, then the 40 runs in three
  waves, about **3.4 hours** in all, about $6.
- **m7i.24xlarge** on-demand, `--jobs 25` for the validation and `--jobs 40` for the runs: one wave
  that runs slower per job, about **2.2 hours**, about $12.
- Memory: 2.5 GB per validation worker, 5.5 GB per run. Storage: 30 GiB is plenty.

On the laptop, from the branch, committed and pushed:

```powershell
& $PY scripts\run_matrix.py --matrix configs\exp_ensemble_k30_matrix.yaml --dry-run | Select-Object -First 1   # 40 jobs
& $PY scripts\make_bundle.py --markets us --out $env:TEMP\afe_bundle_ensemble
scp $env:TEMP\afe_bundle_ensemble\afe_data.tar $env:TEMP\afe_bundle_ensemble\SHA256SUMS $env:TEMP\afe_bundle_ensemble\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7, a smoke test
(2 candidates x 2 seeds at 1 epoch, then each matrix row once):

```bash
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python
$PY scripts/ensemble_validation.py --config configs/us_ensemble_val.yaml --smoke --jobs 4 --run-dir runs/_smoke/ensval   # ends "-> seeds" or "-> configs"
$PY scripts/run_matrix.py --matrix configs/exp_ensemble_k30_matrix.yaml --smoke --jobs 8   # "0 failed"
```

Then the run in tmux, one line (m7i.8xlarge; on the m7i.24xlarge `--jobs 25` and `--jobs 40`):

```bash
tmux new -s afe
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python   # again inside tmux: PY does not carry over
for w in us_ensemble_val us_ensemble_val_fixedwin; do $PY scripts/ensemble_validation.py --config configs/$w.yaml --jobs 16 --threads 2 --run-dir runs/_ensval/$w; done 2>&1 | tee runs/ensval_console.log; $PY scripts/run_matrix.py --matrix configs/exp_ensemble_k30_matrix.yaml --jobs 16 --threads 2 2>&1 | tee runs/matrix_ensemble_console.log; tar -czf ~/runs_ensemble.tgz -C ~/dlsa/attention-factors-europe runs; sudo shutdown -h now
```

Progress from the laptop: `ssh "ubuntu@$IP" "tail -n 3 ~/dlsa/attention-factors-europe/runs/ensval_console.log ~/dlsa/attention-factors-europe/runs/matrix_ensemble_console.log"`.
The validation answer is the last line of `ensval_console.log` for each window, minutes after the
start. Collect as in section 11 (`runs_ensemble.tgz`); the results land in `runs/_ensval/<config>/`
and `runs/_matrix/exp_ensemble_k30`.

## 14. The headline without depositary receipts, K = 30 (2026-11-04)

Taking receipts out of the saved K = 30 books lowered net Sharpe by about 0.55, all of it before 2010, and left
one US run where Europe on the US values stands (`runs/_matrix/exp_ensemble_k30/traded/rebook_no_receipts.py`).
Nothing was retrained there: the models still saw receipts. `configs/exp_no_receipts_matrix.yaml` retrains the
headline, sobol-036 at K = 30 and seeds 0-4, on three universes (15 runs):
- `us_headline`: the baseline on this machine, as in the gap matrices;
- `us_drop_receipts`: the same members without their receipts, slots left empty. Only the receipts change, so
  this is the row to read against `us_headline` and against the re-booked books;
- `us_no_receipts`: the 500 largest listings that are not receipts. Slots refilled with smaller names, so it
  moves for two reasons, and the 800-name row of section 9 says the smaller names matter.

Size, from the K = 30 runs of sections 12 and 13 (63 minutes each at 16 jobs): **m7i.8xlarge** on-demand,
`--jobs 16`, about **1.1 hours**, about $2. Memory 5.5 GB per run, storage 30 GiB.

On the laptop, from the branch, committed and pushed; the two builds are about 15 minutes each and need no WRDS:

```powershell
& $PY scripts\build_us_dataset.py --config configs\us_data_no_receipts.yaml
& $PY scripts\build_us_dataset.py --config configs\us_data_drop_receipts.yaml
& $PY scripts\run_matrix.py --matrix configs\exp_no_receipts_matrix.yaml --dry-run | Select-Object -First 1   # 15 jobs
& $PY scripts\make_bundle.py --markets us us_no_receipts us_drop_receipts --out $env:TEMP\afe_bundle_receipts
scp $env:TEMP\afe_bundle_receipts\afe_data.tar $env:TEMP\afe_bundle_receipts\SHA256SUMS $env:TEMP\afe_bundle_receipts\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7:

```bash
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python
$PY scripts/run_matrix.py --matrix configs/exp_no_receipts_matrix.yaml --smoke --jobs 3   # "0 failed"
tmux new -s afe
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python   # again inside tmux: PY does not carry over
$PY scripts/run_matrix.py --matrix configs/exp_no_receipts_matrix.yaml --jobs 16 --threads 2 2>&1 | tee runs/matrix_no_receipts_console.log
```

Collect as in section 11; the results land in `runs/_matrix/exp_no_receipts`.

## 15. Model v2, the whole attention thesis matrix (2026-11-06)

`configs/thesis_v2_matrix.yaml`: model v2 (`configs/us_headline_v2.yaml`, CLAUDE.md) on every attention row of
the two thesis matrices but the raw-input ones, with the same spec names, plus a ladder of K = 30 rows on the
old values that takes the old headline to v2 one change at a time. Four searches run first (`us_v2`,
`us_v2_lag1`, `eu_v2`, `eu_v2_lag1`); each row but the ladder trades with a winner. 214 runs + 4 searches:

| group | rows | runs |
|---|---|---|
| headline | `us_attention`, `eu_attention`, `eu_attention_usvalues`: K 1-100, seeds 0-4 | 105 |
| ablation | six characteristic groups x two markets, K = 30, seeds 0-2 | 36 |
| robustness | lag 1 x two, unfilled, europe12, US top 200, no receipts, K = 30, seeds 0-2 | 18 |
| extension | 2022-2025 x two, K = 30, seeds 0-4 | 10 |
| ladder | four US steps, five Europe steps, K = 30, seeds 0-4 | 45 |

The PCA rows use no characteristics, so `runs/_matrix/thesis` and `thesis_eu` keep their numbers (with the fixed
window: section 16).

Size: about 167 laptop process-hours by the matrix's cost model; the thesis matrix was about 150 and took 5.2
hours on the m7i.24xlarge at 46 jobs. Expect about **6 hours**, about **$35** on-demand, spot not advised: 6
hours is long to lose. Memory: 5.5 GB per US run, 3 GB per Europe run, 2-2.5 GB per search worker. Storage:
**40 GiB** gp3, encrypted. Results about 3.5 GB, every run keeping its weights (15 MB a US run, 6 MB a Europe run);
`evaluation/diagnostics.py` says what else a run keeps (6 MB at K = 30, more with K, less for Europe's names).

On the laptop, from the branch, committed and pushed. The stage 2c tables take about 5 minutes and need no WRDS:

```powershell
cd C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-model-v2\attention-factors-europe
$PY = "C:\Users\henri\miniconda3\envs\afe\python.exe"
foreach ($c in "us_data", "us_data_no_receipts", "europe17_data", "europe12_data") { & $PY scripts\build_asof_tables.py --config configs\$c.yaml }
& $PY scripts\run_matrix.py --matrix configs\thesis_v2_matrix.yaml --dry-run | Select-Object -First 1   # 218 jobs
& $PY scripts\make_bundle.py --markets us europe us_no_receipts us_no_receipts_asof --out $env:TEMP\afe_bundle_v2
scp $env:TEMP\afe_bundle_v2\afe_data.tar $env:TEMP\afe_bundle_v2\SHA256SUMS $env:TEMP\afe_bundle_v2\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7, a smoke test (each search on
2 points at 1 epoch, then every spec once):

```bash
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python
$PY scripts/run_matrix.py --matrix configs/thesis_v2_matrix.yaml --smoke --jobs 16   # "0 failed"
```

Then the run in tmux, one line, which also packs the results and stops the machine:

```bash
tmux new -s afe
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python   # again inside tmux: PY does not carry over
$PY scripts/run_matrix.py --matrix configs/thesis_v2_matrix.yaml --jobs 46 --threads 2 2>&1 | tee runs/matrix_v2_console.log; tar -czf ~/runs_v2.tgz -C ~/dlsa/attention-factors-europe runs; sudo shutdown -h now
```

Progress from the laptop: `ssh "ubuntu@$IP" "tail -n 3 ~/dlsa/attention-factors-europe/runs/matrix_v2_console.log"`.
Look once after the first hour: the four winners are in `runs/_matrix/thesis_v2/searches/*/metrics.json`, and a net
Sharpe above 10 on validation means something broke. A restart of the same command skips what finished.

Collect as in section 11 (`runs_v2.tgz`, then terminate the machine); the results land in
`runs/_matrix/thesis_v2`. `summary_by_spec.csv` has every row; the ladder is the group `ladder`, read top to
bottom, and its last US step against `us_attention` is what the new search did. Run directories on the machine
hold machine paths in the ledger: map them by basename, as before. Full factor weights on any date of a run:
`python scripts/factor_replay.py runs/<run> --dates 2008-10-01`.

## 16. The PCA rows under model v2 (2026-11-06)

`configs/thesis_v2_pca_matrix.yaml`: every PCA row of the two thesis matrices with the fixed residual window and
cumulative input, so that attention and its benchmarks trade on one set of mechanics (the file's header says why
the other two changes of model v2 do not reach PCA). Same spec names, six new searches, a ladder. 167 runs + 6
searches + 2 stage-one builds, 175 jobs:

| group | rows | runs |
|---|---|---|
| headline | `us_pca_longconv`, `eu_pca_longconv`, `eu_pca_longconv_usvalues`: K 1-100, seeds 0-4 | 120 |
| headline | `us_pca_ou`, `eu_pca_ou`: K 1-100, one deterministic run | 16 |
| robustness | lag 1 x two, K = 30, seeds 0-2 | 6 |
| ladder | two US steps, three Europe steps, K = 30, seeds 0-4 | 25 |

Size: 41 laptop process-hours by the matrix's cost model, against 167 for the v2 matrix, which took 5.6 hours on the
m7i.24xlarge at 46 jobs. Expect **2 to 3 hours**, about **$15** on-demand. The stage ones rebuild on the machine:
6 GB each, 10 and 4 GB of memory. Memory: 3 GB per US run, 2 GB per Europe run. Storage: **40 GiB** gp3, encrypted.
Results about 2 GB. No data of its own: the bundle of section 15 is enough, only `COMMIT` changes.

On the laptop, from the branch, committed and pushed:

```powershell
cd C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-model-v2\attention-factors-europe
$PY = "C:\Users\henri\miniconda3\envs\afe\python.exe"
& $PY scripts\run_matrix.py --matrix configs\thesis_v2_pca_matrix.yaml --dry-run | Select-Object -First 1   # 175 jobs
& $PY scripts\make_bundle.py --markets us europe --out $env:TEMP\afe_bundle_v2_pca
scp $env:TEMP\afe_bundle_v2_pca\afe_data.tar $env:TEMP\afe_bundle_v2_pca\SHA256SUMS $env:TEMP\afe_bundle_v2_pca\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7, a smoke test (each search on
2 points at 1 epoch, then every spec once):

```bash
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python
$PY scripts/run_matrix.py --matrix configs/thesis_v2_pca_matrix.yaml --smoke --jobs 16   # "0 failed"
```

Then the run in tmux, one line, which also packs the results and stops the machine:

```bash
tmux new -s afe
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python   # again inside tmux: PY does not carry over
$PY scripts/run_matrix.py --matrix configs/thesis_v2_pca_matrix.yaml --jobs 46 --threads 2 2>&1 | tee runs/matrix_v2_pca_console.log; tar -czf ~/runs_v2_pca.tgz -C ~/dlsa/attention-factors-europe runs; sudo shutdown -h now
```

Look once after the first hour: the six winners are in `runs/_matrix/thesis_v2_pca/searches/*/metrics.json`. Collect
as in section 11 (`runs_v2_pca.tgz`, then terminate the machine). The ladder is the group `ladder`, read top to
bottom: step 0 against the October rows is the machine noise, step 1 is what the fixed window did, and the last
step against the headline row is what the new search did. Read it before calling the rows like for like: B_t is fitted
on a span containing the window, so PCA's fixed window is not the same experiment as attention's
(docs/pca_longconv_benchmark.md).

## 17. The Table 4 search, searches only (2026-11-08)

`configs/us_search_t4.yaml` and `configs/europe_search_t4.yaml`: the v2 searches (section 15) with every tuning
parameter of the paper's Table 4 searched as well, instead of held at the paper's values. Same window (fit
1990-1995 US, 1993-1995 Europe, net Sharpe on 1996-1997), model v2 held, nothing out of sample. 15 dimensions:

| setting | range | paper |
|---|---|---|
| hidden dim d (`policy.hidden`) | 16, 32, 64 | 32 |
| dropout | 0, 0.1, 0.2, 0.3 | 0.1 |
| attention dim d_x (`model.embedding_dim`) | 16, 32, 64 | 32 |
| epochs | 10, 20, 30, 45, 60 | 30 |
| LongConv layers | 1, 2 | 1 |
| lambda_VAR | 0.1-300, log | 100 |
| learning rate | 0.0003-0.03, log | 0.003 |
| weight decay (LongConv) | 0.005-0.5, log | 0.05 |
| LongConv init | geometric decay, random | geometric decay |
| lambda_squash | 0.0001-0.01, log | 0.001 |
| input scale, score temperature, ridge, batch days, window_vol | as in section 15 | not in the paper |

512 Sobol points on seed 0, then the best 10, the paper's Table 4 (`base`) and the v2 winners (`us_v2_headline`,
`eu_v2_headline`, named finalists) on seeds 0-2; each Europe search also carries its US winner. Every winner lands
in `runs/_matrix/search_t4/searches/<name>/selected.yaml` and `searches.json`. No run trades with them: that is a
later matrix, after a decision.

Two matrices, one out dir:

- `configs/search_t4_k30_matrix.yaml`: K = 30, US then Europe, 46 workers each.
- `configs/search_t4_allk_matrix.yaml`: one search per K of Table 2 (1, 3, 5, 8, 15, 30, 100) and market, two at a
  time with 23 workers each. Same name and the same two K = 30 searches as the K = 30 matrix, so it can run after
  it and skip them.

Size, m7i.24xlarge on-demand ($5.14/h) at 46 workers: K = 30 only about **7.6 hours** (US 6.4, then Europe 1.2;
1 in 10 draws of the finalists about 9), about **$40**; every K about **47 hours**, about **$240**, or about 40 more
hours after a K = 30 run. The searches of different K are independent: on several machines the wall-clock falls,
the cost does not. These are estimates: laptop timings of each setting on the US first window
(`policy.layers`, hidden, attention dim, K; 2026-11-08), scaled by the 195 s (US) and 38 s (Europe) a v2 trial took on
this machine (section 15); the deeper stack may lose more than that to a fully loaded machine, so allow a third
more.

The cost is the LongConv layers: a deeper stack runs every one of the 30 window positions through each layer but
the last, where one layer reads only the last one, so a 2-layer epoch costs 8.5x a one-layer one at hidden 32 and
16x at hidden 64. The mean US trial is about 25 minutes (v2: 3), the longest (2 layers, hidden 64, 60 epochs) about
3 hours; trials start longest first, so those do not end a search. Three layers (not searched: the paper's value
and double, as for d and d_x) would cost about 65% more; with one layer only the search would take 1.2 hours.

Memory: 2.5 GB per worker plus up to 4.4 GB for a deeper stack (above that it recomputes its pieces in backward);
the matrix budgets 6.5 GB per US worker and 3 GB per Europe worker.
Storage: **30 GiB** gp3, encrypted. Results are small (trials.jsonl, under 50 MB).

On the laptop, from the branch, committed and pushed (the data is the bundle of section 16, only `COMMIT` changes):

```powershell
cd C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-table4\attention-factors-europe
$PY = "C:\Users\henri\miniconda3\envs\afe\python.exe"
& $PY scripts\run_matrix.py --matrix configs\search_t4_k30_matrix.yaml --dry-run    # 2 jobs (allk: 14)
& $PY scripts\make_bundle.py --markets us europe --out $env:TEMP\afe_bundle_t4
scp $env:TEMP\afe_bundle_t4\afe_data.tar $env:TEMP\afe_bundle_t4\SHA256SUMS $env:TEMP\afe_bundle_t4\COMMIT "ubuntu@${IP}:~"
```

On the machine, after the clone, `sha256sum -c` and `cloud_setup.sh` of section 7, a smoke test (each search on
2 points, the base and the named finalists, 1 epoch):

```bash
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python
$PY scripts/run_matrix.py --matrix configs/search_t4_k30_matrix.yaml --smoke --jobs 8   # "0 failed"
```

Then the run in tmux (`search_t4_allk_matrix.yaml` for every K), which also packs the results and stops the machine:

```bash
tmux new -s afe
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python   # again inside tmux: PY does not carry over
$PY scripts/run_matrix.py --matrix configs/search_t4_k30_matrix.yaml --jobs 46 --threads 2 2>&1 | tee runs/matrix_t4_console.log; tar -czf ~/runs_t4.tgz -C ~/dlsa/attention-factors-europe runs/_matrix/search_t4; sudo shutdown -h now
```

On Spot, use section 18 instead of the tmux line: a one-time Spot machine loses the search with its disk.
The other K on GPUs: section 19.

Progress from the laptop: `ssh "ubuntu@$IP" "tail -n 2 ~/dlsa/attention-factors-europe/runs/_matrix/search_t4/logs/search_us_t4_K30.log"`
(one line per finished trial, `[search i/513]`). A restart of the same command resumes every search from its
trials.jsonl. Collect as in section 11 (`runs_t4.tgz`, then terminate the machine). Read the finalists table at the
end of each search log: the paired difference of the winner against `base` (Table 4) and against the v2 winner on
the same three seeds is what searching Table 4 bought on validation.

## 18. Spot for a long matrix: persistent request, resumed at every boot (2026-11-08)

A one-time Spot instance is terminated on interruption and its disk goes with it, so a 7-hour search would start
over. A **persistent** Spot request with interruption behaviour **Stop** keeps the disk instead: AWS stops the
machine (two minutes' notice) and starts it again when capacity returns. `scripts/spot_resume.sh install` makes
the matrix part of the boot: a systemd service starts `run_matrix.py` now and at every boot until the matrix has
finished. A restarted matrix skips finished jobs and each search resumes from its `trials.jsonl`, so an
interruption costs the trials in flight (the longest go first, so early on that can be up to ~3 hours of the deep
trials) plus the time without capacity. A log line cut off by the stop is dropped and rerun (`runs.read_jsonl`).
At the end the service packs `runs/_matrix/<name>` and the console log into `~/<name>.tgz`, leaves
`~/<name>.done` and shuts the machine down; a later boot then does nothing.

Launch as in section 2 with the instance type of the matrix (m7i.24xlarge for section 17; m6i.24xlarge or
m7a.24xlarge also have the 384 GiB its memory budget assumes), and under Advanced details:

| field | value |
|---|---|
| Purchasing option | **Spot instances** -> Customize |
| Request type | **Persistent** |
| Interruption behavior | **Stop** |
| Request valid until | a few days ahead, so a forgotten request lapses |
| Maximum price | leave empty (capped at the on-demand price) |
| Shutdown behavior | Stop (the default) |

Optional: EC2 -> Elastic IPs -> Allocate, then Associate with the instance, so the address survives a stop
(about $0.005/h). Without it, read the new public IP in the console after each restart.
Price and interruption frequency: EC2 -> Spot Requests -> Pricing history, and the Spot Instance Advisor.

Send the bundle, clone, unpack and set up as in section 7, run the smoke test of the matrix, then on the machine,
instead of the tmux line:

```bash
cd ~/dlsa/attention-factors-europe
bash scripts/spot_resume.sh install configs/search_t4_k30_matrix.yaml --jobs 46 --threads 2
bash scripts/spot_resume.sh status                     # "active (running)"
tail -f runs/search_t4_console.log                     # one "=== ... start or resume" line per boot
```

When it is done the machine is stopped. To collect: EC2 -> Instance state -> Start (the service sees the `.done`
file and does nothing), then on the laptop `scp "ubuntu@${IP}:~/search_t4.tgz" $env:TEMP\` and unpack as in
section 11. Then **cancel the Spot request** (EC2 -> Spot Requests -> Actions -> Cancel request, tick "terminate
instances") and check that no volume is left (EC2 -> Volumes). A stop from inside the machine counts as your own
stop, so AWS does not restart it; the request still has to be cancelled, or a stopped instance keeps its disk
(and its cost) until it lapses.

## 19. The Table 4 search at every K, on GPUs (2026-11-09)

The all-K matrix of section 17 is about 40 more hours on the m7i.24xlarge after the K = 30 run, almost all of it
in the 2-layer LongConv trials: large dense operations, which a GPU does far faster than a CPU. So the searches
can train on GPUs: `scripts/search_validation.py --device cuda`, through `scripts/run_matrix.py --device cuda`
(searches only; runs and PCA prep stay on the CPU and a matrix with any to do refuses the flag).
`configs/search_t4_allk_gpu_matrix.yaml` is `search_t4_allk_matrix.yaml` with 8 workers a search and less host
memory a worker: same fourteen searches, same names, same out dir. Unpack the K = 30 results of section 17 into
the repo first and the two K = 30 searches are skipped: the twelve others run (US K = 100, 15, 8, 5, 3, 1, each
Europe search after its US K).

What runs on the GPU and what does not:
- Each worker loads the 1990-1997 panel on the CPU and keeps a copy on one GPU (worker i on GPU i mod the number
  of GPUs); the training steps and the validation spans run there, and the net, turnover and short series come
  back to the CPU to be scored as before. Matmuls stay float32 (TF32 off).
- The initial weights are drawn on the CPU, so they are those of the CPU trial. The dropout masks come from the
  GPU's generator and the float sums run in another order, so a GPU trial is **not** the CPU trial number for
  number. A CPU trial already moves by about 0.25 net Sharpe between the laptop and AWS (2026-11-08); what has to
  hold is the ranking. `scripts/gpu_probe.py` checks both before the run.
- A deeper LongConv runs in pieces of 8192 windows on a GPU (1024 on the CPU) and recomputes them in backward
  above 2^26 hidden values (2^27 on the CPU), since several workers share a GPU. Each process gets at most
  0.85 / (processes per GPU) of the GPU's memory (`--gpu-procs`, which run_matrix sets to `--jobs`); a trial that
  runs out of it is run again from the start with every piece recomputed (same numbers, slower) and its line in
  trials.jsonl says `"oom_retry": true`. Every GPU trial records its `device`.
- The CPU path is unchanged: a CPU smoke search gives the same trials, digit for digit, as commit 9e780db.

**Before anything else: GPU quota.** New accounts usually have none, and the request can take hours to days.
Service Quotas -> Amazon EC2 -> **All G and VT Spot Instance Requests** (for Spot) or **Running On-Demand G and VT
instances**: 48 vCPUs for a 4-GPU machine (4 for a 1-GPU one). G instances may not be offered in Stockholm: if
the launch page does not list them, use Ireland (eu-west-1) or Frankfurt (eu-central-1), and import the
`henrik-laptop` public key there (EC2 -> Key pairs -> Actions -> Import key pair).

Machine: **g6.12xlarge** (4 x NVIDIA L4 24 GB, 48 vCPU, 192 GiB) or **g5.12xlarge** (4 x A10G 24 GB, same CPU and
memory). AMI: **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**, which has the NVIDIA driver; torch
comes from PyPI with its own CUDA libraries. Storage: the AMI's default size or at least 60 GiB gp3 (the CUDA
build of torch is about 5 GB), encrypted. Spot as in section 18 (persistent, interruption behaviour Stop).
For only a probe, a g6.xlarge (1 x L4, 4 vCPU, 16 GiB) is enough.

None of the GPU speeds have been measured yet. The probe measures them on the machine, and its numbers give the
time and the number of workers per GPU; until then, the guess of 2026-11-08 stands: a few hours on 4 GPUs.

On the laptop, from the branch, committed and pushed (same data as section 17, plus the K = 30 results):

```powershell
cd C:\Users\henri\Desktop\SSE\masterThesis\Code\dlsa-table4\attention-factors-europe
$PY = "C:\Users\henri\miniconda3\envs\afe\python.exe"
& $PY scripts\run_matrix.py --matrix configs\search_t4_allk_gpu_matrix.yaml --dry-run --jobs 16   # 12 jobs
& $PY scripts\make_bundle.py --markets us europe --out $env:TEMP\afe_bundle_t4gpu
scp $env:TEMP\afe_bundle_t4gpu\afe_data.tar $env:TEMP\afe_bundle_t4gpu\SHA256SUMS $env:TEMP\afe_bundle_t4gpu\COMMIT "ubuntu@${IP}:~"
scp runs\_incoming\search_t4.tgz "ubuntu@${IP}:~/search_t4_k30_cpu.tgz"
```

On the machine (clone and `sha256sum -c` as in section 7), then:

```bash
cd ~/dlsa/attention-factors-europe; PY=~/afe-venv/bin/python
bash scripts/cloud_setup.sh --gpu                       # lists the GPUs, ends with "cuda 12... ['NVIDIA L4', ...]"
tar -xzf ~/search_t4_k30_cpu.tgz                        # runs/_matrix/search_t4: the two K = 30 searches
$PY -m pytest -q tests/test_gpu_search.py               # the CUDA tests run here: "8 passed", none skipped
$PY scripts/gpu_probe.py --search runs/_matrix/search_t4/searches/eu_t4_K30 --stage step
$PY scripts/gpu_probe.py --search runs/_matrix/search_t4/searches/us_t4_K30 --jobs 4
```

Read the probe (`runs/<stamp>_gpu_probe_*/metrics.json`):
- **step**: every line `ok` (relative differences of float32 rounding, around 1e-5). `CHECK` means the GPU
  computes something else: stop.
- **time**: seconds an epoch and peak memory of the heaviest setting with its pieces kept and recomputed. Workers
  per GPU: about 0.85 x GPU memory / (peak reserved + 0.5 GB of context), and no more than the vCPUs allow (one
  core a worker); then `--jobs` = workers per GPU x GPUs.
- **replay**: 16 trials of the K = 30 search plus its winner, on the GPU against the CPU's record. Spearman rho
  near 1 and differences of the size of the machine noise (about 0.25), not a shift of the whole ranking.

Then a smoke run of the matrix, and the run itself as in section 18, with `--device cuda`:

```bash
$PY scripts/run_matrix.py --matrix configs/search_t4_allk_gpu_matrix.yaml --smoke --jobs 4 --device cuda   # "0 failed"
bash scripts/spot_resume.sh install configs/search_t4_allk_gpu_matrix.yaml --jobs 16 --threads 1 --device cuda
bash scripts/spot_resume.sh status                      # "active (running)"
nvidia-smi                                              # every GPU busy, --jobs processes in all
```

`--jobs 16` on 4 GPUs is two searches at a time with 8 workers each, 4 processes a GPU; set it from the probe.
Progress as in section 17 (`runs/_matrix/search_t4/logs/search_us_t4_K100.log`, one line per trial). At the end
the service packs `~/search_t4.tgz` (all fourteen searches, K = 30 included) and stops the machine; collect it and
cancel the Spot request as in section 18.
