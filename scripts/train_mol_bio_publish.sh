#!/bin/bash
set -e

python train_mol_bio_publish.py | tee ../logs/train_mol_bio_publish.log
python eval_mol_bio_publish.py | tee ../logs/eval_mol_bio_publish.log
