TRAIN_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_train_0_edited.csv
VAL_DIR=/home/zchayav/projects/syntheye/datasets/eye2gene_new_filepaths/all_baf_valid_50deg_filtered_val_0_edited.csv
SAVE_DIR=experiment_3/
DEVICE=1
python train_ae.py --train-fpath=$TRAIN_DIR --val-fpath=$VAL_DIR --latent-size=1024 --loss=mae --epochs=200 --device=$DEVICE --save-dir=$SAVE_DIR --save-step=10