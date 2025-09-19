#!/bin/bash
#SBATCH -J train_drugs
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH --gpus=a100:1
#SBATCH -t 00:30:00
#SBATCH --array=1-5
#SBATCH -o logs/train_drugs/%a.out

TEST_FOLD=$SLURM_ARRAY_TASK_ID
MAX_STEPS=${MAX_STEPS:=1000}
EXPERIMENT_NAME=${EXPERIMENT_NAMEL="test-drugs"}
DEVICE="cuda"

python train.py \
  config/train_delphi-drugs.py \
  --data data/transforms/drugs/all.bin \
  --delphi_labels data/transforms/drugs/delphi_labels_chapters_colours_icd_with_drugs.csv \
  --test_fold $TEST_FOLD \
  --experiment-name ${EXPERIMENT_NAME} \
  --max_steps=${MAX_STEPS} \
  --device=${DEVICE} \
  --eval_iters=100 \
  --auc \
  "$@"
