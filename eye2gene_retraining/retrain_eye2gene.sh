TRAIN_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_train_0_edited.csv
VAL_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_val_0_edited.csv
SAVE_DIR=one_fold_retraining/noise_3600_only_again
python retrain.py --model=inceptionv3 --train=$TRAIN_DIR --val=$VAL_DIR --epochs=100 --resize=299 --batch-size=128 --lr=0.0001 --weight-decay=0 --save-best=1 --w-monitor=acc --es=1 --es-monitor=loss --save-dir=$SAVE_DIR --save-logs --device=3 --verbose --seed=1502