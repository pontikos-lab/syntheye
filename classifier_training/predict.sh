#!/usr/bin/bash
TEST_DIR=/home/zchayav/projects/syntheye/datasets/syntheye/testset_just_5_classes.csv
MODEL_DIR=trained_models/vit_5classes_rebalanced
WEIGHTS=best_weights.pth
CONFIG=model_config.json
DEVICE=2
python predict.py --config=$MODEL_DIR/$CONFIG --weights=$MODEL_DIR/$WEIGHTS --test=$TEST_DIR --save-dir=$MODEL_DIR --verbose --device=$DEVICE --seed=1399