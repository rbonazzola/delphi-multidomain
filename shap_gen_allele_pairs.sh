source ~/repos/codon_helpers/slurm_functions.sh

EXPERIMENT_ID=263078128312970150
MIN_SUBJECTS=${MIN_SUBJECTS:-300}
MAX_SUBJECTS=${MAX_SUBJECTS:-30000}
TOKEN_METADATA="data/transforms/tokens/diseases/token_metadata.tsv"
SUBJECTS=${SUBJECTS:-data/transforms/subject_lists/genetic_white_ids.txt}

PARAMS_TSV=$(mktemp /tmp/shap_gen_pairs_XXXX.tsv)
awk -F'\t' -v min="$MIN_SUBJECTS" -v max="$MAX_SUBJECTS" '
  NR == 1 { print "disease_id\tpairs_output"; next }
  $5 >= min && $5 <= max { print $1 "\t" "shap/pairs_" $3 ".tsv" }
' "$TOKEN_METADATA" > "$PARAMS_TSV"

n=$(( $(wc -l < "$PARAMS_TSV") - 1 ))
echo "Submitting pair generation for $n diseases (MIN=${MIN_SUBJECTS}, MAX=${MAX_SUBJECTS})"

sarray_params shap/custom_hla_shap_v2.py "$PARAMS_TSV" \
  --experiment_id ${EXPERIMENT_ID} \
  --generate_pairs \
  --min_pair_freq 0.005 \
  --run_name "hla_mix" \
  --param n_head=12 \
  --subjects "${SUBJECTS}" \
  --time=01:00:00 --mem=16G --cpus=2 \
  "$@"

rm -f "$PARAMS_TSV"
