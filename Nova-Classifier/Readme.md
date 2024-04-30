# Single-Cell Classification of Protein Labels, Mechanism of Actions (MoAs), and Other Relevant Predictors

This repository contains code for single-cell classification of protein labels, mechanism of actions (MoAs), and other relevant predictors using various deep learning models.

## Table of Contents

1. [Hardware Configuration](#hardware-configuration)
2. [Software Setup](#software-setup)
3. [Data Preprocessing](#data-preprocessing)
4. [Full Training](#full-training)

## Hardware Configuration

### Training Mode

- **Run Training:**
  ```bash
  !python main.py train -i save_directory -j options/nova.yaml
  ```
- Update configuration file setup for custom model options, including directory paths to training, validation, and testing csv's.
- Initialize separate ```basic_train``` and ```\dataloaders\datasets``` for different datasets. Key parameters to be adjusted include: 1) # of labels to predict, 2) normalization values and other custom transform functions, 3) fixed number of cells to load in per image.
- Specify model class (including EfficienNet, Resnet, MedViTs) in ```\models``` folder, can implement additional model classes.
- Note: Training will save model weights of best epoch checkpoint, keep a running log of losses and metrics: mean average precision (mAP) and area under the curve (AUC), and save final predictions (weighted image-level and cell-level predictions) and corresponding ground truth as csv's in the ```results``` directory.
- Note: MAE pretraining & subsequent ViT finetuning separately trained, not using weighted image-cell level dual head predictor skeleton. Please refer to Nova-Classifier/mae.ipynb notebook for training details.

### Testing mode
* ```!python main.py test -i save_directory -j options/nova.yaml```
* Generates gradcam in ```log``` directory of training results folder for every class.
* Outputs class predictions
* Calculates metrics (mAP) for the best model checkpoint on the testing dataset, which is chosen by the highest mAP during training.

## II. SOFTWARE
* To run this codebase, please configure packages to the versions in ```novaenv_requirements.txt```.
* Training supported on GPUs P100, V100, A40, and higher versions.

## III. Data Preprocessing

1. []

2. []

### Full training
* []
