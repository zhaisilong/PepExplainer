#!/bin/bash

set -e

for seed in {0..9}; do
    python train_mol_bio_good.py --seed $seed | tee ../logs/train_mol_bio_good_${seed}.log
    python eval_mol_bio_good.py --seed $seed | tee ../logs/eval_mol_bio_good_${seed}.log
done
