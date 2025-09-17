TEST_FOLD=${TEST_FOLD:=1}
MAX_STEPS=${MAX_STEPS:=1000}
EXPERIMENT_NAME=${EXPERIMENT_NAMEL="test-drugs"}
DEVICE="cpu"

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
