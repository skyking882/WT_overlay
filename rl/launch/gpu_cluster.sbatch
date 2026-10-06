#!/bin/bash
# GPU cluster node: 1 GPU + 60 CPU cores. Submit from the repository root:
#     sbatch rl/launch/gpu_cluster.sbatch
# Environment overrides: RL_RUN_DIR (default $HOME/rl_runs/gpu_cluster), RL_PYPY (PyPy for the env workers),
#     RL_PYTHON (python with torch, default: python), RL_EXTRA ("--set ppo.clip=0.1 ...").
#SBATCH --job-name=wt-rl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=%x-%j.out

set -u
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# module load python cuda      # site specific: a python with torch (CUDA build) must be on PATH
PY="${RL_PYTHON:-python}"
RUN_DIR="${RL_RUN_DIR:-$HOME/rl_runs/gpu_cluster}"
mkdir -p "$RUN_DIR"

SETS=()
if [ -n "${RL_PYPY:-}" ]; then SETS+=(--set "workers.python=['$RL_PYPY']"); fi
# all 60 cores are ours: 56 env workers + learner threads (see rl/config.py preset gpu_cluster)

# The training process checkpoints every ckpt_every rounds and on SIGUSR1 / SIGTERM (10 min before the
# time limit, see --signal above). It exits with 75 when it stopped early and wants to be run again.
"$PY" -m rl.train run --config gpu_cluster --run-dir "$RUN_DIR" ${SETS[@]+"${SETS[@]}"} ${RL_EXTRA:-} &
pid=$!
trap 'kill -USR1 "$pid" 2>/dev/null' USR1 TERM
wait "$pid"; rc=$?
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done
echo "rl.train exit code $rc"
if [ "$rc" -eq 75 ]; then
    # unfinished: continue from the checkpoint in a new job
    scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || sbatch "$0"
fi
exit 0
