#!/bin/bash
# GPU smoke test for the 10-layer diseases-stream / 1-layer HLA-stream / 2-layer
# trunk DelphiMultiStream config validated locally in this session.
# Small subject subset + few epochs: only checks it runs on a real GPU
# (dtype/memory/torch.compile behavior), not meant to produce a usable model.
python train_multi_stream.py \
  --arch "MultiStream([diseases,death,lifestyle,sex]:(h12d240l10),[hla_alleles,genetic_pcs,sex]:(h12d240l1)):(h12d240l2)" \
  --attention_scheme "[hla_alleles,sex,genetic_pcs]:bidirectional,all:causal(mask_ties=True)" \
  --domains diseases,death,lifestyle,sex,hla_alleles,genetic_pcs \
  --experiment_name test_crossattn \
  --run_name 10A_1B_2trunk_gpu \
  --max_epochs 300 \
  --patience 10 \
  --batch_size 128 \
  --no-compile \
  --block_size 96 \
  --no_event_token_rate 5 \
  --compute_aucs \
  --no_rich \
  "$@"
  
# --subjects data/transforms/subject_lists/10000.csv \
