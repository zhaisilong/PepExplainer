#!/bin/bash
set -e
task=thrombin
pushd ..
for seed in {0..9}; do
    python train_mol_${task}.py --seed $seed | tee logs/train_mol_${task}_${seed}.log
    python eval_mol_${task}.py --seed $seed | tee logs/eval_mol_${task}_${seed}.log
done
popd
