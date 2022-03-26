TRAIN_DIR=/home/zchayav/projects/syntheye/image_encoder/ae_trainset.csv
VAL_DIR=/home/zchayav/projects/syntheye/image_encoder/ae_testset.csv
SAVE_DIR=experiment_7/
DEVICE="0,1"
python train_ae.py --train-fpath=$TRAIN_DIR --val-fpath=$VAL_DIR --resize=512 --latent-size=2048 --loss=mae --epochs=100 --device=$DEVICE --save-dir=$SAVE_DIR --save-step=10