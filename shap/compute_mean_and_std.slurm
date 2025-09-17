#!/bin/bash
#SBATCH -J shap_mean_std          # Job name
#SBATCH -o logs/%A_%a.out         # Standard output
#SBATCH -e logs/%A_%a.err         # Standard error
#SBATCH --time=02:00:00           # Time limit
#SBATCH --cpus-per-task=4         # CPUs per task
#SBATCH --mem=128g                 # Memory
#SBATCH --array=0-9               # Job array indices (10 files)

# List of input files
FILES=(
  shap_values_merged_356a27d0f1314c3cb0441705e9e8a431.pkl
  shap_values_merged_5e79356b24de4d7badd9ae7e2ec55da8.pkl
  shap_values_merged_b2571198bf7d470e8a213f1f01a80f6b.pkl
  shap_values_merged_dc74321948ca409fafd57c80e476b2d8.pkl
  shap_values_merged_4047c0cc96db483ab7f09f11e43f17dd.pkl
  shap_values_merged_818f0b764eaa4ec9a22b9f26e0ac3ec4.pkl
  shap_values_merged_bcd76052049b41ac905747d3ab59f2c1.pkl
  shap_values_merged_5a61b289a7a24c429d27b1963293cac3.pkl
  shap_values_merged_8985fa4dcc254d2eb825beda333a6a29.pkl
  shap_values_merged_dc052a634b0d4580b590c19bfe494691.pkl
)

INPUT_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "Processing file: $INPUT_FILE"
python compute_mean_and_std.py "output/${INPUT_FILE}"

