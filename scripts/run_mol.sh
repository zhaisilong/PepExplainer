#!/bin/bash
set -e

pushd ..
for seed in {0..9}; do
    python train_mol.py --seed $seed | tee logs/train_mol_${seed}.log
    python eval_mol.py --seed $seed | tee logs/eval_mol_${seed}.log
done
popd
