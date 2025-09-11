TEST_FOLD=1

python train.py \
  config/train_delphi-drugs.py \
  --data data/transforms/drugs/all.bin \
  --test_fold $TEST_FOLD \
  --experiment-name testcito \
  --max_steps=100 \
  --eval_iters=100 \
  "$@"
