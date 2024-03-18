#!/bin/bash
set -e

for cv in {0..4}; do
    python train_mask_bio_publish_cv.py --cv $cv | tee ../logs/train_mask_bio_publish_${cv}.log
    python eval_mask_bio_publish_cv.py --cv $cv | tee ../logs/eval_mask_bio_publish_${cv}.log
done
