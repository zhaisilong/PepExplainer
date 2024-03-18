#!/bin/bash
set -e
python train_mol.py | tee ../logs/train_mol.log
python eval_mol.py | tee ../logs/eval_mol.log
