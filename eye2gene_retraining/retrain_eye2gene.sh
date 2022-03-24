TRAIN_DIR=/home/zchayav/projects/syntheye/datasets/syntheye/all_folds/real+stylegan2_rebalanced.csv
VAL_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_val_0_edited.csv
SAVE_DIR=simple_convnet/rebalanced
DEVICE=3
python retrain.py --model=simple-convnet --train=$TRAIN_DIR --val=$VAL_DIR --normalize=0 --n-channels=1 --epochs=100 --resize=512 --batch-size=128 --lr=0.0001 --weight-decay=0 --save-best=1 --w-monitor=acc --es=1 --es-monitor=loss --save-dir=$SAVE_DIR --save-logs --device=$DEVICE --verbose --seed=1399