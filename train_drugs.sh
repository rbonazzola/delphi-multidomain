TEST_FOLD=${TEST_FOLD:=1}
MAX_STEPS=${MAX_STEPS:=1000}
EXPERIMENT_NAME="test-drugs"

python train.py \
  config/train_delphi-drugs.py \
  --data data/transforms/drugs/all.bin \
  --test_fold $TEST_FOLD \
  --experiment-name ${EXPERIMENT_NAME} \
  --max_steps=${MAX_STEPS} \
  --eval_iters=100 \
  "$@"
