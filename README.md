## Team nova-networks
# Investigation of Weakly Supervised Multi-label Image Classifier for Microscopy Image Representation 

Phenotypic representation for microscopy images has been demonstrated as a powerful method for research in cell biology, which can also serve as a tool for profiling perturbations in drug discovery. However, achieving a single-cell level of cellular annotations is often challenging due to the vast scale of compound screening. Deep learning methods have been proposed as a solution to encode high-quality image representations that both recapitulate features of the datasets and provide explainable information, despite often being difficult to interpret. Here, we propose a framework that combines the results of weakly supervised learning (WSL) with Class Activation Mapping (CAM) to enhance explainability. We test our idea using the 2021 Human Protein Atlas Kaggle Challenges to build our testing framework for crafting a multi-channel, multi-label classifier. This framework integrates a Cellpose-based single cell segmentation, an image preprocessor, and various network architectures to provide plug-and-play finetuning. For model inspection, we leverage Gradient-weighted Class Activation Mapping (Grad-CAM) as the primary tool to visualize the focus of our model during the inference phase, thereby enhancing the explainability of model performance. We subsequently apply this framework to the Broad Bioimage Benchmark Collection (BBBC) datasets to understand the representation on Cell Painting images associated with the mechanism of action (MoA) in molecules. Our aim is to expedite drug development by providing a deep learning-based phenotypic representation that aligns closely with experimental design, while still offering sufficient explainability for scientific decision-making.

Our final code is developed in the *Nova-Classifier* directory.

# Code specifics
We developed a model skeleton trainable on the HPA dataset and BBBC036 dataset. After preprocessing and segmentation with Cellpose, we applied data normalization and batch processing to conduct model training. We trained on EfficientNet, Resnet, ViT, MedViT, and ViT finetuning with MAE pretraining.
To run basic training workflow, please run the following command. Input argument ```-i``` specifies directory that will store the contents of this command, in ```/results/```. Input argument ```-i``` specifies configuration file (YAML file). ```main-training.ipynb``` and ```main-testing.ipynb``` provide an example for training and testing modalities.
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

# Model Workflow
![alt text](https://github.com/sunni426/nova-networks/blob/main/modified_team2_pipeline.png?raw=true)

# Acknowledgements
We are deeply grateful to our advisors from Novartis for mentoring us in this project.
