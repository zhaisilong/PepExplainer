#!/bin/bash
set -e

for seed in {0..9}; do
    python train_mask.py --seed $seed | tee ../logs/train_mask_${seed}.log
    python eval_mask.py --seed $seed | tee ../logs/eval_mask_${seed}.log
done
