#!/bin/bash
set -e

for seed in {0..9}; do
    python train_mask_bio_publish.py --seed $seed | tee ../logs/train_mask_bio_publish_${seed}.log
    python eval_mask_bio_publish.py --seed $seed | tee ../logs/eval_mask_bio_publish_${seed}.log
done
