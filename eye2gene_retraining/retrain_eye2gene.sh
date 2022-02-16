TRAIN_DIR=/home/zchayav/projects/syntheye/synthetic_datasets/dummy_synthetic_100/generated_examples.csv
VAL_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_val_0_edited.csv
SAVE_DIR=noise_3600_only
python retrain.py --model=inceptionv3 --train=$TRAIN_DIR --val=$VAL_DIR --epochs=100 --resize=299 --batch-size=256 --lr=0.0001 --weight-decay=0 --save-best=1 --w-monitor=acc --es=1 --es-monitor=loss --save-dir=$SAVE_DIR --save-logs --device=1,3 --verbose --seed=1399