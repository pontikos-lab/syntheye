# Model Training with Synthetic Data

This sub-directory contains scripts to train deep learning models with our ophthalmic datasets.

Requirements:
- `pytorch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`

A model can be trained by modifying the `train.sh` script and then executing on the terminal. The training set and validation set must be provided as csv files containing two columns - `"file.path"` for the filepath and `"gene"` containing the class. The model can also be trained on one or multiple GPUs by specifying device number(s) below. To do this, put in a single number in quotes (`DEVICE="0"`) if one device, or comma-separated numbers (`DEVICE="0,1,2,3"`) if multiple devices. The model choice can be one of the following: `["inceptionv3", "simple-convnet", "densenet-169", "resnet50"]`

```
TRAIN_DIR=/my/train/data/csv
VAL_DIR=/my/val/data/csv
SAVE_DIR=/save/to/this/dir
DEVICE=0
python retrain.py --model=simple-convnet --train=$TRAIN_DIR --val=$VAL_DIR --normalize=0 --n-channels=1 --epochs=100 --resize=512 --batch-size=128 --lr=0.0001 --weight-decay=0 --save-best=1 --w-monitor=acc --es=1 --es-monitor=loss --save-dir=$SAVE_DIR --save-logs --device=$DEVICE --verbose --seed=1399
```

To generate results for the trained model, modify the `predict.sh` script and then execute in the terminal. Again, the test data is provided as csv with same structure as train/validation set. The model directory is the same as in `SAVE_DIR` above. During training, model weights are by default saved as `best_weights.pth`, so add that in the `WEIGHTS` below, unless it was renamed later. Single and Multi-GPU running is also supported by specifying device id.

```
TEST_DIR=/my/test/data/csv
MODEL_DIR=/my/saved/model/dir
WEIGHTS=/name/of/best/weights/file
CONFIG=/path/to/model/config/
DEVICE=0
python predict.py --config=$MODEL_DIR/$CONFIG --weights=$MODEL_DIR/$WEIGHTS --test=$TEST_DIR --save-dir=$MODEL_DIR --verbose --device=$DEVICE --seed=1399
```