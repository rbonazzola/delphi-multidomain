source ~/repos/codon_helpers/slurm_functions.sh

EXPERIMENT_ID=263078128312970150
MIN_SUBJECTS=${MIN_SUBJECTS:-10000}
MAX_SUBJECTS=${MAX_SUBJECTS:-30000}
TOKEN_METADATA="data/transforms/tokens/diseases/token_metadata.tsv"
SUBJECTS=${SUBJECTS:-data/transforms/subject_lists/genetic_white_ids.txt}

awk -F'\t' -v min="$MIN_SUBJECTS" -v max="${MAX_SUBJECTS:-inf}" \
  'NR > 1 && $5 >= min && $5 <= max { print $1 "\t" $2 }' "$TOKEN_METADATA" | \
while IFS=$'\t' read -r DISEASE_ID DISEASE; do
  ICD10=$(awk '{print $1}' <<< "${DISEASE}")
  PAIRS_TSV="shap/pairs_${ICD10}.tsv"

  if [[ ! -f "$PAIRS_TSV" ]]; then
    echo "Skipping ${DISEASE}: pairs file not found (${PAIRS_TSV})"
    continue
  fi

  echo "Submitting: ${DISEASE} (disease_id=${DISEASE_ID}, pairs=$(( $(wc -l < "$PAIRS_TSV") - 1 )))"

  sarray_params shap/custom_hla_shap_v2.py "$PAIRS_TSV" \
    --experiment_id ${EXPERIMENT_ID} \
    --disease_id ${DISEASE_ID} \
    --run_name "hla_mix" \
    --param n_head=12 \
    --n_counterfactuals 5 \
    --subjects "${SUBJECTS}" \
    --time=06:00:00 --mem=32G --cpus=8 \
    "$@"
done
