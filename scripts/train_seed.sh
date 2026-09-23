#!/bin/bash
# Task 3: train TiRGN ICEWS14 with a fixed seed on a given GPU.
# Hyperparameters: see configs/task3_icews14.md (identical to README baseline + --seed).
# Usage: bash scripts/train_seed.sh <seed> <gpu>
set -e
SEED=$1
GPU=$2
cd "$(dirname "$0")/../src"
source /newSSD/MrLiu/miniconda3/etc/profile.d/conda.sh
conda activate tirgn4090
PYTHONPATH=.. LD_LIBRARY_PATH=$CONDA_PREFIX/lib python main.py \
  -d ICEWS14 --history-rate 0.3 --train-history-len 9 --test-history-len 9 \
  --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 \
  --self-loop --decoder timeconvtranse --encoder convgcn --layer-norm \
  --weight 0.5 --entity-prediction --relation-prediction --add-static-graph \
  --angle 14 --discount 1 --task-weight 0.7 --gpu "$GPU" --save checkpoint \
  --seed "$SEED" 2>&1 | tee "../logs/train_ICEWS14_seed${SEED}_gpu${GPU}.log"
