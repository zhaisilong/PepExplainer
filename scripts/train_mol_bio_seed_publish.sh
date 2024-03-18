#!/bin/bash

set -e

for seed in {0..9}; do
    python train_mol_bio_publish.py --seed $seed | tee ../logs/train_mol_bio_publish_${seed}.log
    python eval_mol_bio_publish.py --seed $seed | tee ../logs/eval_mol_bio_publish_${seed}.log
done
