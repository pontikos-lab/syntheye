TEST_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_val_0_edited.csv
MODEL_DIR=noise_3600_only
WEIGHTS=best_weights.pth
CONFIG=model_config.json
python predict.py --config=$MODEL_DIR/$CONFIG --weights=$MODEL_DIR/$WEIGHTS --test=$TEST_DIR --resize=299 --normalize=1 --save-dir=$MODEL_DIR --verbose --device=1 --seed=1399