#!/bin/bash
set -e

python train_mol_bio_good.py | tee ../logs/train_mol_bio_good.log
python eval_mol_bio_good.py | tee ../logs/eval_mol_bio_good.log
