#!/bin/bash

set -e

python train_mask_bio.py | tee ../logs/train_mask_bio.log
python eval_mask_bio.py | tee ../logs/eval_mask_bio.log