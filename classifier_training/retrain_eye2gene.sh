#!/bin/bash

MODEL=inceptionv3
TRAIN_DIR=/home/zchayav/projects/syntheye/classifier_training/datasets/RB_r1.csv
VAL_DIR=/home/zchayav/projects/syntheye/dataset/all_baf_valid_50deg_filtered_val_0.csv
SAVE_DIR=experiments/RB_r1/
DEVICE=2,3
python retrain.py --model=$MODEL --train=$TRAIN_DIR --val=$VAL_DIR --normalize=1 --n-channels=3 --epochs=100 --resize=299 --batch-size=128 --lr=0.0001 --weight-decay=0 --save-best=1 --monitor=loss --save-dir=$SAVE_DIR --save-logs --device=$DEVICE --verbose --seed=1399