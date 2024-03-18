#!/bin/bash

set -e

# python train_mask.py | tee ../logs/train_mask.log
python eval_mask.py | tee ../logs/eval_mask.log