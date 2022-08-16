#!/bin/bash
TEST_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_val_0_edited.csv
MODEL_DIR=/home/zchayav/projects/syntheye/classifier_training/trained_models/inception_model/extra_experiments/synthetic_with_same_real_dist/
WEIGHTS=best_weights.pth
CONFIG=model_config.json
DEVICE=3
python predict.py --config=$MODEL_DIR/$CONFIG --weights=$MODEL_DIR/$WEIGHTS --test=$TEST_DIR --save-dir=$MODEL_DIR --verbose --device=$DEVICE --seed=1399