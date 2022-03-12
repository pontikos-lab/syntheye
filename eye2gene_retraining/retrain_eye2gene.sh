TRAIN_DIR=/home/zchayav/projects/syntheye/datasets/syntheye/all_folds/synthetic_10800_only.csv
VAL_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_val_0_edited.csv
SAVE_DIR=simple_model/all_folds_retraining/synthetic_10800_only
DEVICE=3
python retrain.py --model=simple --train=$TRAIN_DIR --val=$VAL_DIR --normalize=1 --n-channels=3 --epochs=100 --resize=299 --batch-size=128 --lr=0.0001 --weight-decay=0 --save-best=1 --w-monitor=acc --es=1 --es-monitor=loss --save-dir=$SAVE_DIR --save-logs --device=$DEVICE --verbose --seed=1399