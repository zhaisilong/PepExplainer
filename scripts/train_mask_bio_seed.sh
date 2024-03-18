#!/bin/bash
set -e

for seed in {0..9}; do
    python train_mask_bio.py --seed $seed | tee ../logs/train_mask_bio_${seed}.log
    python eval_mask_bio.py --seed $seed | tee ../logs/eval_mask_bio_${seed}.log
done
