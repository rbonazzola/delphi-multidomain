#!/bin/bash
#SBATCH --job-name=hla_shap
#SBATCH --array=0-358
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=5
#SBATCH --output=/nfs/research/birney/users/bonazzola/repos/delphi/shap/logs/hla_shap_%A_%a.out

ALLELE_ID=${SLURM_ARRAY_TASK_ID}

cd /nfs/research/birney/users/bonazzola/repos/delphi

python shap/custom_hla_shap.py \
    --allele_id ${ALLELE_ID} \
    --disease "${DISEASE}"
