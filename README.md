# nova-networks

Team 2 Kaggle https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238645
Team 2 Base code: https://github.com/iseekwonderful/HPA-singlecell-2nd-dual-head-pipeline

To run basic training workflow, please run the following command. Input argument ```-i``` specifies directory that will store the contents of this command, in ```/results/```. Input argument ```-i``` specifies configuration file (YAML file)
```
!python main.py train -i t259 -j jakiro/sin_exp5_b3_rare.yaml # -i: results dir name
```
For custom configurations:
> configs/jakiro/sin_exp5_b3_rare.yaml (For specific configurations. Take care of paths)
> 
> configs/__init__.py (For all other default configurations)

Ground truth files in:
> dataloaders/split

Ground truth files are stored as .csv files in:
> dataloaders/split

Please use the packages in the following requirements file for virtual environment configuration:
> novaenv_requirements.txt

# Model Workflow
![alt text](https://github.com/sunni426/nova-networks/blob/main/modified_team2_pipeline.png?raw=true)
