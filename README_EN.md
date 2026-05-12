# Experiment Running Guide

This document only describes the required dependencies and the experiment workflow.

## 1. Dependencies

Python 3.7+ is recommended.

Main dependencies:

```text
numpy
joblib
tqdm
torch
matplotlib
```

If any dependency is missing, install it with:

```bash
pip install numpy joblib tqdm torch matplotlib
```

## 2. Raw Data Preparation

The project requires the following two raw data files:

```text
data/NYC.npy
data/NYC_POI.npy
```

`data/NYC_data.pkl` is a generated preprocessing cache. It is large and does not need to be uploaded. It can be regenerated on the server:

```bash
python load.py --dname NYC
```

After this step, the following file will be generated:

```text
data/NYC_data.pkl
```

## 3. Low-Frequency POI Filtering Experiments

These experiments evaluate how different low-frequency POI filtering thresholds affect the recommendation results.

Generate the `min20` dataset:

```bash
python load.py \
  --dname NYC \
  --output_dname NYC_min20 \
  --min_loc_visits 20 \
  --min_user_checkins 6
```

Generate the `min30` dataset:

```bash
python load.py \
  --dname NYC \
  --output_dname NYC_min30 \
  --min_loc_visits 30 \
  --min_user_checkins 6
```

Generate the `min50` dataset:

```bash
python load.py \
  --dname NYC \
  --output_dname NYC_min50 \
  --min_loc_visits 50 \
  --min_user_checkins 6
```

The generated files will be saved under `data/`. For example:

```text
data/NYC_min20.npy
data/NYC_min20_POI.npy
data/NYC_min20_data.pkl
```

## 4. Running Experiments With Slurm

This project provides a Slurm script:

```text
run_stan_collab.sh
```

Before running it, set the project path according to your own server environment. It is recommended to pass the path through environment variables when submitting the job instead of hardcoding personal paths in the script.

### 4.1 Required Path

You must set:

```bash
PROJECT_DIR=/path/to/your/project
```

`/path/to/your/project` should be the directory containing this project. It should include:

```text
load.py
train_neighbor.py
collab_common.py
run_stan_collab.sh
data/
```

For example, if the project is located at `/your/project/path`, submit a job with:

```bash
PROJECT_DIR=/your/project/path sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

You can optionally set an output directory:

```bash
RUN_DIR=/path/to/your/project/test
```

If `RUN_DIR` is not set, the script will use:

```bash
${PROJECT_DIR}/test
```

### 4.2 Conda Environment

The script uses the following defaults:

```bash
CONDA_ROOT=$HOME/miniconda3
CONDA_ENV=ai_study
```

If your conda path or environment name is different, specify them when submitting:

```bash
PROJECT_DIR=/your/project/path \
CONDA_ROOT=/path/to/miniconda3 \
CONDA_ENV=your_env_name \
sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

### 4.3 Slurm Options to Check

The script starts with the following Slurm settings:

```bash
#SBATCH --job-name=stan_collab
#SBATCH --partition=gbunchQ
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
```

Please check or modify the following options according to your server:

```text
--partition       partition name
--time            maximum running time
--mem             memory
--cpus-per-task   number of CPUs
--gres=gpu:1      number of GPUs
```

If your server does not use the `gbunchQ` partition, change:

```bash
#SBATCH --partition=gbunchQ
```

to your available partition, for example:

```bash
#SBATCH --partition=gpu
```

### 4.4 Submit Experiments

Submit the `NYC_min20` experiment:

```bash
PROJECT_DIR=/path/to/your/project sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

Submit the `NYC_min30` experiment:

```bash
PROJECT_DIR=/path/to/your/project sbatch run_stan_collab.sh neighbor NYC_min30 0 30
```

Submit the `NYC_min50` experiment:

```bash
PROJECT_DIR=/path/to/your/project sbatch run_stan_collab.sh neighbor NYC_min50 0 30
```

Argument meanings:

```text
neighbor   use the neighbor collaborative variant
NYC_min20  dataset name
0          use all users
30         train for 30 epochs
```

If you do not want to pass `PROJECT_DIR=...` every time, export it first:

```bash
export PROJECT_DIR=/path/to/your/project
export RUN_DIR=/path/to/your/project/test
sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

## 5. Common Hyperparameter Settings

Hyperparameters can be changed through environment variables.

Change the learning rate:

```bash
LR=5e-4 sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

Change the collaborative term weight:

```bash
COLLAB_WEIGHT=0.3 sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

Clip the collaborative term `C`:

```bash
C_CLIP=1.0 sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

Change multiple parameters at the same time:

```bash
LR=5e-4 COLLAB_WEIGHT=0.3 C_CLIP=1.0 sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

## 6. What the Code Outputs

### 6.1 Output of `load.py`

When running preprocessing or filtering, for example:

```bash
python load.py --dname NYC --output_dname NYC_min20 --min_loc_visits 20 --min_user_checkins 6
```

the terminal prints filtering statistics:

```text
clean_stats: {
  'raw_checkins': number of raw check-ins,
  'raw_users': number of raw users,
  'raw_locs': number of raw POIs,
  'clean_checkins': number of check-ins after filtering,
  'clean_users': number of users after filtering,
  'clean_locs': number of POIs after filtering,
  'dropped_checkins': number of removed check-ins,
  'min_loc_visits': minimum POI visit threshold,
  'min_user_checkins': minimum user check-in threshold
}
```

It also prints numeric progress messages, for example:

```text
100 85
200 13
...
0
100
200
...
```

These numbers are preprocessing progress logs for user trajectories and the POI distance matrix.

After preprocessing, files will be generated under `data/`, for example:

```text
data/NYC_min20.npy
data/NYC_min20_POI.npy
data/NYC_min20_data.pkl
```

### 6.2 Output of the Training Script

After submitting a Slurm job, for example:

```bash
sbatch run_stan_collab.sh neighbor NYC_min20 0 30
```

Slurm prints the job id:

```text
Submitted batch job 208405
```

The training log is saved to:

```text
/path/to/your/project/test/JOBID_stan_collab.log
```

The error log and progress bar output are saved to:

```text
/path/to/your/project/test/JOBID_stan_collab.err
```

The `.log` file first prints environment information:

```text
HOSTNAME
DATE
Python version
CUDA check
torch version
cuda available
device 0
```

Then it prints the experiment configuration:

```text
VARIANT=neighbor
DNAME=NYC_min20
PART=0
EPOCHS=30
LR=1e-3
COLLAB_WEIGHT=0.5
C_CLIP=0
```

During training, it periodically prints statistics for the collaborative term `C`:

```text
[C-STATS] epoch:1, fstep:1000, split:train, mask_len:21,
c_abs_mean:..., c_abs_max:..., weighted_c_abs_mean:...,
base_abs_mean:..., weighted/base_mean:...
```

These values are used to check whether the collaborative term `C` is too large. If `C` is too large, the log will contain:

```text
[C-WARN]
```

At the end of each epoch, the script prints training loss and validation/test metrics:

```text
epoch:1, time:..., train_loss:..., train_ce_loss:..., train_aux_loss:...
epoch:1, time:..., valid_acc:[R@1 R@5 R@10 R@20]
epoch:1, time:..., test_acc:[R@1 R@5 R@10 R@20]
```

The metric order is:

```text
valid_acc[0] = valid Recall@1
valid_acc[1] = valid Recall@5
valid_acc[2] = valid Recall@10
valid_acc[3] = valid Recall@20

test_acc[0] = test Recall@1
test_acc[1] = test Recall@5
test_acc[2] = test Recall@10
test_acc[3] = test Recall@20
```

When validation Recall@5 improves, the best checkpoint is saved and the log prints:

```text
best_r5:0.xxxxxx, saved:best_neighbor_stan_NYC_min20.pth
```

After training, the model checkpoint is:

```text
best_neighbor_stan_NYC_min20.pth
```

The `.err` file mainly contains the `tqdm` progress bar, for example:

```text
50%|█████     | 542/1083 [29:43<27:12,  3.02s/it]
```

If the job times out or fails, Slurm error messages will also appear in the `.err` file.
