#!/bin/bash

MODEL=inceptionv3
TRAIN_DIR=/home/zchayav/projects/syntheye/datasets/syntheye/five_classes_rebalanced.csv
VAL_DIR=/home/zchayav/projects/syntheye/datasets/syntheye/testset_just_5_classes.csv
SAVE_DIR=trained_models/inceptionv3_5classes_rebalanced/
DEVICE=0,1
python retrain.py --model=$MODEL --train=$TRAIN_DIR --val=$VAL_DIR --normalize=1 --n-channels=3 --epochs=100 --resize=299 --batch-size=128 --lr=0.0001 --weight-decay=0 --save-best=1 --monitor=loss --save-dir=$SAVE_DIR --save-logs --device=$DEVICE --verbose --seed=1399