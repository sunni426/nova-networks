# Team Nova-Networks
## Investigation of Weakly Supervised Multi-label Image Classifier for Microscopy Image Representation

Phenotypic representation for microscopy images has been demonstrated as a powerful method for research in cell biology, which can also serve as a tool for profiling perturbations in drug discovery. However, achieving a single-cell level of cellular annotations is often challenging due to the vast scale of compound screening. Deep learning methods have been proposed as a solution to encode high-quality image representations that both recapitulate features of the datasets and provide explainable information, despite often being difficult to interpret. Here, we propose a framework that combines the results of weakly supervised learning (WSL) with Class Activation Mapping (CAM) to enhance explainability. We test our idea using the 2021 Human Protein Atlas Kaggle Challenges to build our testing framework for crafting a multi-channel, multi-label classifier. This framework integrates a Cellpose-based single cell segmentation, an image preprocessor, and various network architectures to provide plug-and-play finetuning. For model inspection, we leverage Gradient-weighted Class Activation Mapping (Grad-CAM) as the primary tool to visualize the focus of our model during the inference phase, thereby enhancing the explainability of model performance. We subsequently apply this framework to the Broad Bioimage Benchmark Collection (BBBC) datasets to understand the representation on Cell Painting images associated with the mechanism of action (MoA) in molecules. Our aim is to expedite drug development by providing a deep learning-based phenotypic representation that aligns closely with experimental design, while still offering sufficient explainability for scientific decision-making.

### Code specifics
We developed a model skeleton trainable on the HPA dataset and BBBC036 dataset. After preprocessing and segmentation with Cellpose, we applied data normalization and batch processing to conduct model training. We trained on EfficientNet, Resnet, ViT, MedViT, and ViT finetuning with MAE pretraining.

### Model Workflow
![alt text](https://github.com/sunni426/nova-networks/blob/main/modified_team2_pipeline.png?raw=true)

# Single-Cell Classification of Protein Labels, Mechanism of Actions (MoAs), and Other Relevant Predictors

This repository contains code for single-cell classification of protein labels, mechanism of actions (MoAs), and other relevant predictors using deep learning models on different cellular datasets. Our final code is developed in the ```Nova-Classifier``` directory.

## Table of Contents

1. [Software Setup](#software-setup)
2. [Data Preprocessing](#data-preprocessing)

## Software Setup

### Training Mode

To run basic training workflow, please run the following command. Input argument ```-i``` specifies directory that will store the contents of this command, in ```results```. Input argument ```-j``` specifies configuration file (YAML file). ```main-training.ipynb``` and ```main-testing.ipynb``` provide an example for training and testing modalities.
```
!python main.py train -i save_directory -j options/nova.yaml
```
For custom configurations:
> configs/options/nova.yaml (For specific configurations, including ground truth and input paths, training configs)
> 
> configs/__init__.py (For all other default configurations)

Ground truth csv files are stored as .csv files in:
> dataloaders/split

Please use the packages in the following requirements file to configure virtual environment configuration:
> novaenv_requirements.txt

- **Run Training:**
  ```bash
  !python main.py train -i save_directory -j options/nova.yaml
  ```
- Update configuration file setup for custom model options, including directory paths to training, validation, and testing csv's.
- Initialize separate ```basic_train``` and ```dataloaders/datasets``` for different datasets. Key parameters to be adjusted include:
  - Number of labels to predict
  - Normalization values and other custom transform functions
  - Fixed number of cells to load in per image
- Specify model class (including EfficienNet, Resnet, MedViTs) in ```models``` folder, can implement additional model classes.
- **Note**: Training utilities:
  - Save model weights of best epoch checkpoint
  - Keep a running log of losses and metrics: mean average precision (mAP) and area under the curve (AUC)
  - Save final predictions (weighted image-level and cell-level predictions) and corresponding ground truth as csv's in the ```results``` directory.
- **Note**: MAE pretraining & subsequent ViT finetuning separately trained, not using weighted image-cell level dual head predictor skeleton. Please refer to ```Nova-Classifier/mae.ipynb``` notebook for training details.

- The two datasets we trained on are:
  - Human Protein Atlas (HPA), https://www.proteinatlas.org/
  - Broad Bioimage Benchmark Collection (BBBC036) Human U2OS Cells – profiling bioactive compounds using Cell Painting, https://bbbc.broadinstitute.org/BBBC036

### Testing mode
- **Run Testing:**
  ```bash
  !python main.py test -i save_directory -j options/nova.yaml
  ```
- Generates gradcam in ```log``` directory of training results folder for every class.
- Outputs class predictions
- Calculates metrics (mAP) for the best model checkpoint on the testing dataset, which is chosen by the highest mAP during training.

### Version Compatibility
* To run this codebase, please configure packages to the versions in ```novaenv_requirements.txt```.
* Training supported on GPUs P100, V100, A40, and higher versions.

## Data Preprocessing

To run preprocessing for the HPA and BBBC datasets, please follow the instructions provided in the ```preproc_bbbc.ipynb``` and ```preproc_HPA.ipyn``` in the ```preprocessing``` folder.
![alt text](https://github.com/sunni426/nova-networks/blob/main/Nova-Classifier/preprocessing.png?raw=true)

### Acknowledgements
We are deeply grateful to our advisors from Novartis Discovery Sciences Data Science Biomedical Research for mentoring us in this project.

*@Boston University, Department of Biomedical Engineering, 2024*
