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
