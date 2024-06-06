#!/bin/bash
set -e
task=hla
pushd ..
for seed in {1..9}; do
    python train_mask_${task}.py --seed $seed | tee logs/train_mask_${task}_${seed}.log
    python eval_mask_${task}.py --seed $seed | tee logs/eval_mask_${task}_${seed}.log
done
popd
