#!/bin/bash

set -e

for cv in {0..4}; do
    python train_mol_bio_publish.py --cv $cv | tee ../logs/train_mol_bio_publish_${cv}.log
    python eval_mol_bio_publish.py --cv $cv | tee ../logs/eval_mol_bio_publish_${cv}.log
done
