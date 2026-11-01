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
goes with them, finished years included. Runs are short, so relaunch and start over. If
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
