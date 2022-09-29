#!/bin/bash
TEST_DIR=/home/zchayav/projects/stylegan2-ada-pytorch/synthetic_datasets/stylegan2_synthetic_-2perclass/generated_examples2.csv
MODEL_DIR=/home/zchayav/projects/syntheye/classifier_training/experiments/R_r1/
SAVE_DIR=/home/zchayav/projects/syntheye/classifier_training/experiments/synthetic_data_evaluation
WEIGHTS=best_weights.pth
CONFIG=model_config.json
DEVICE=cpu
python predict.py --config=$MODEL_DIR/$CONFIG --weights=$MODEL_DIR/$WEIGHTS --test=$TEST_DIR --save-dir=$SAVE_DIR --verbose --device=$DEVICE --seed=1399