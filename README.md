<div style="text-align: justify">

# SynthEye

Inherited Retinal Diseases (IRDs) represent a diverse group of rare monogenic diseases resulting in visual impairment and blindness. Over 200 different types of IRDs are known, which affect 1 in 3000 people in the UK. Genetic diagnosis is particularly relevant in IRD care, however remains elusive in more than 40% of cases due to the lack of clinical experience and inefficiencies in the diagnostic services.

IRDs have specific patterns that can be identified using retinal imaging. The Moorfield's Eye Hospital (MEH) IRD cohort contains retinal images and their corresponding genetic diagnoses for more than 3,000 families in the UK. However, there is a significant amount of class imbalance, as some genetic diseases are more common than others. **SynthEye** aims to address this imbalanced through synthetic data augmentation with Generative Adversarial Networks (GANs). This repository contains code for the implemented methods in our paper.

## Dependencies

Experiments can be run in a conda virtual environment (Python 3.9.5) with the following libraries installed:

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

For our experiments, we used 3 NVIDIA GeForce 3090 RTX GPUs, each with 25 GB memory.

## Generating Synthetic Data

We have trained a StyleGAN2-ADA model for generating our synthetic IRD data. Please see the following repo from NVIDIA-Labs for details: [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch). Most of the hyperparameters are the recommended values because conducting further hyperparameter tuning is computationally time-consuming.

Previously, experiments were conducted with multi-scale gradient GANs (MSGGANs) (see [here](https://github.com/akanimax/BMSG-GAN)), which we modified to generate FAF images of 36 different IRDs given a noise vector and a class index. The models are in `legacy/models/`. Examples of results from these experiments are saved in `legacy/images/`.

## Deep Learning Model Training with Synthetic Data

The second part of our study trains a deep convolutional network with novel datasets composed of real and synthetic data. We use an [InceptionV3 classifier](https://arxiv.org/abs/1512.00567). The models and experiments are shared in `classifier_training/`.

</div>