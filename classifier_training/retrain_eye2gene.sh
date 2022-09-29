#!/bin/bash

MODEL=inceptionv3
TRAIN_DIR=/home/zchayav/projects/syntheye/all_baf_valid_50deg_filtered_train_0_edited.csv
VAL_DIR=/home/zchayav/projects/syntheye/dataset/all_baf_valid_50deg_filtered_val_0.csv
SAVE_DIR=experiments/feature_extractor
DEVICE=2,3
python retrain.py --model=$MODEL --train=$TRAIN_DIR --val=$VAL_DIR --normalize=1 --n-channels=3 --epochs=100 --resize=512 --batch-size=64 --lr=0.0001 --weight-decay=0 --save-best=1 --monitor=loss --save-dir=$SAVE_DIR --save-logs --device=$DEVICE --verbose --seed=1399