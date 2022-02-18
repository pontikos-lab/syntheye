TRAIN_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_train_0_edited.csv
VAL_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_val_0_edited.csv
SAVE_DIR=all_folds_retraining/baseline
python retrain.py --model=simple --train=$TRAIN_DIR --val=$VAL_DIR --epochs=100 --resize=299 --batch-size=512 --lr=0.0001 --weight-decay=0 --save-best=1 --w-monitor=acc --es=1 --es-monitor=loss --save-dir=$SAVE_DIR --save-logs --device=0 --verbose --seed=1399