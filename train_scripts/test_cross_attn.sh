#!/bin/bash
python train_cross_attn.py \
  --arch "CrossAttention([hla_alleles,sex,genetic_pcs]:(h24d240l3),[diseases,lifestyle,sex]:(h24d240l6),xattn:(h24d240)):(h24d240l6)" \
  --attention_scheme "[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)" \
  --domains diseases,lifestyle,death,genetic_pcs,hla_alleles,sex \
  --dcfg diseases.predict=True \
  --experiment_name test_crossattn \
  --max_epochs 1000 \
  --patience 10 \
  --batch_size_schedule "10:64,10:128,5:256,*:256x4" \
  --no-compile \
  "$@"
