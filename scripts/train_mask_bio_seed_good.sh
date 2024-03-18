#!/bin/bash
set -e

for seed in {0..9}; do
    python train_mask_bio_good.py --seed $seed | tee ../logs/train_mask_bio_good_${seed}.log
    python eval_mask_bio_good.py --seed $seed | tee ../logs/eval_mask_bio_good_${seed}.log
done
