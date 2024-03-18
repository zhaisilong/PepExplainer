#!/bin/bash

set -e

python train_mask_bio_publish.py | tee ../logs/train_mask_bio_publish.log
python eval_mask_bio_publish.py | tee ../logs/eval_mask_bio_publish.log