#!/bin/bash
set -e

python train_mol_bio.py | tee ../logs/train_mol_bio.log
python eval_mol_bio.py | tee ../logs/eval_mol_bio.log
