<div style="text-align: justify">

# SynthEye

Inherited Retinal Diseases (IRDs) represent a diverse group of rare monogenic diseases resulting in visual impairment 
and blindness. Over 200 different types of IRDs are known, which affect 1 in 3000 people in the United Kingdom. 
Genetic diagnosis is particularly relevant in IRD management, however remains elusive in more than 40% of cases due to 
the lack of clinical experience and inefficiencies in the diagnostic services.

IRDs have specific progression patterns that can be identified using retinal imaging modalities like 
fundus autofluorescence (FAF), infrared imaging, and optical coherence tomography. The Moorfield's Eye Hospital (MEH) 
IRD cohort contains retinal images and their corresponding genetic diagnoses for more than 3,000 families in the UK. An 
ongoing project pursued by our team, Eye2Gene, uses deep learning to genetically diagnose 36 IRDs from retinal images 
and the existing models were trained on the MEH IRD dataset. However, there is a significant amount of class imbalance, 
since some genetic diseases are less common than others. To tackle the imbalance, this project **SynthEye** implements 
Generative Adversarial Networks (GANs) that can synthesize new images of the specific IRD mutated genes. This repository
contains code for executing our models. 

## Dependencies

We ran our experiments in a conda virtual environment with Python 3.9.5. The following packages must be installed:
- PyTorch == 1.8.1
- Torchvision == 0.2.2 
- NumPy == 1.20.2
- Pandas == 1.2.5
- ScikitLearn == 0.24.2
- SciPy == 1.6.2
- TensorBoard == 2.5.0
- Seaborn == 0.11.1
- MatplotLib ==  3.3.4
- PyYAML == 0.2.5
- tqdm == 4.59.0

For our experiments, we used 3 NVIDIA GeForce 3090 RTX GPUs.

## Available Models

The models are provided in ``models/msggan/``. We implemented a conditional version of the Multi-scale Gradients GAN (MSGGAN) (see [here](https://github.com/akanimax/BMSG-GAN)), 
which generates FAF images of 36 different IRDs given a noise vector and a class encoding. The noise vector is sampled
from a 512-dimensional Gaussian distribution. For the class encoding, we experimented with 2 methods - one-hot-encodings 
similar to the CGAN implementation (see [here](https://arxiv.org/abs/1411.1784)), and embeddings. We refer to the former
as **CMSGGAN-1** and the latter as **CMSGGAN-2**. The code for these models is in ``models/msggan/conditional_msggan.py`` and 
``models/msggan/conditional_msgganv2.py``. 

With these models, we also experimented with two different losses provided in `models/msggan/losses.py`- the 
Wasserstein loss with Gradient Penality (WGAN-GP) (see [here](https://arxiv.org/abs/1704.00028)) and the 
Relativistic Average Hinge loss (RAHinge) (see [here](https://arxiv.org/abs/1807.00734)).

## How to Train or Test a model

In order to train or test a model, the configuration file `model_configs.yaml` is essential. This file contains fields
for the model inputs and training/testing (hyper)parameters. The fields are described in detail in the block below.

To train a model, enter the fields for `# DATA SETTINGS`, `# GAN I/O SETTINGS` and `# TRAINING SETTINGS` in the above 
config file. Then just run ``python train.py``. 

The trained GAN model can be evaluated by generating some new images and comparing them to the real dataset images. To
do this, fill in the entries for `# TESTING SETTINGS` in the config file. If comparing with the real dataset's images,
make sure the fields under `# DATA SETTINGS` and `GAN I/O SETTINGS` are also entered.

```buildoutcfg
# DATA SETTINGS
data_file: # type string; provide the path to a csv file which contains one column for the image filepaths and a second column for the class of that image 
filenames_col: # type string; column in data_file which contains path to image file
labels_col: # type string; column containing label of the image
train_classes: # type string or list; set to "all" if training on all classes, or provide a list/text file of desired classes to train on
transformations:
  resize_dim: # resolution to resize training set images
  random_flip: # type bool (1/0); performs random flip with p=0.3
  grayscale: # type bool (1/0); convert image to 1 channel grayscale
  normalize: # type bool (1/0); normalizes image across channels

# GAN I/O SETTINGS
model: # type string; model name. Can be ["cmsggan", "cmsgganv2"]
z_dim: # type int; latent space dimension
output_im_resolution: # type int; resolution you want to generate final images at. Should be equal to resize_dim in previous section

# TRAINING SETTINGS
epochs: # type int; number of epochs
loss_fn: # type string; loss function can be ["wgan-gp", "RAHinge"]
batch_size: # type int; batch size
n_disc_updates: # type int; number of discriminator updates
gen_lr: # type float; learning rate for generator
disc_lr: # type float; learning rate for discriminator
beta1: # type float; beta1 parameter for Adam optimizer
beta2: # type float; beta2 parameter for Adam optimizer
display_step: # type int; number of steps to visualize outputs during training
calc_fid: # type bool (1/0); calculates Frechet Inception Distance during training to check performance?
n_samples_to_generate: # type int; number of image samples to visualize in tensorboard during training.
save_checkpoint_steps: # type int; saves model checkpoints every N iterations (batch iterations, not epochs!) 
save_tensorboard: # type bool (1/0); save the tensorboard logging file?
save_weights: # type bool (1/0); save final weights after training?
parallel: # type bool (1/0); Run on multiple GPUs?
device_ids: # type list; specify GPU device ids 

# TESTING SETTINGS
weights_dir: # type string; path to directory of GAN weights/checkpoint files
weights_path: # type string; filename for checkpoint file
gen_classes: # type string or list; set to "all" for evaluating on all classes, or provide a list/text file of classes to evaluate on
n_test_samples: # type int; number of test samples to generate per class
evaluate:
  compute_similarity: # type bool (1/0); compute image similarity between generated set and real dataset
  fid_imagenet: # type bool (1/0); compute standard FID metric on generated images and real images
  fid_eye2gene: # type bool (1/0); compute our version of FID using Eye2Gene weights
  class_preds_eye2gene: # type bool (1/0); compute confusion matrix for Eye2Gene predictions
save_images:
  as_individual: # type bool (1/0); save indivitual images
  most_similar: # type bool (1/0); save most similar generated and real image pairs from the image similarity computation?
  most_different: # type bool (1/0); save most different generated and real image pairs from the image similarity computation?
```

## Sample Results from our model

**CMSGGAN-1** with RAHinge loss was found to be produce the most realistic and biologically correct images. 
The image below shows how the MSGGAN model learns - it starts by learning the lower resolution of the image 
and then grows to higher resolutions.

![alt text](images/cmsggan1_resolutions.gif)

Below are some of the generated images at 256x256 resolution for 8 sample IRD genes.

![alt text](images/cmsggan_1_images.png)

## Interpolation Experiment

Using **CMSGGAN-1**, we performed an interpolation experiment to explore the GAN latent space in 
``notebooks/GAN_Exploration.ipynb``. We interpolate images between two classes by holding the noise vector same but 
modifying the class encoding as shown below:

**Transforming from *ABCA4* --> *BEST1*:**

![alt text](images/ABCA4_2_BEST1.gif)

We also interpolate between two images in the same class keeping the class encoding same but modifying the noise vector.

**Transforming from z1 *ABCA4* --> z2 *ABCA4*:**

![alt text](images/ABCA4_l1_2_l2.gif)

</div>