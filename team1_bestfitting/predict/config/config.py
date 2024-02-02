import os
ope = os.path.exists
opj = os.path.join
import numpy as np
import socket
import warnings
warnings.filterwarnings('ignore')
from argparse import Namespace

sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hostname = socket.gethostname()
DIR_CFGS = Namespace()
DIR_CFGS.RESULT_DIR     = "result" # 'novanetworks/kaggle_HPA/2021/data/HPA_SingleCellClassification_Analysis/team1_bestfitting/result'
DIR_CFGS.MODEL_DIR     = 'novanetworks/kaggle_HPA/2021/data/HPA_SingleCellClassification_Analysis/team1_bestfitting/models'
DIR_CFGS.FEATURE_DIR     = 'novanetworks/kaggle_HPA/2021/data/HPA_SingleCellClassification_Analysis/team1_bestfitting/features'
#'novanetworks/kaggle_HPA/2021/data/HPA_SingleCellClassification_Analysis/team1_bestfitting/features'
DIR_CFGS.DATA_DIR       = '../../../../../2021/data/kaggle-dataset/CAM_images'
DIR_CFGS.PRETRAINED_DIR = '../../../../../2021/HPA_SingleCellClassification_Analysis/team1_bestfitting/pretrained'

NUC_MODEL = f'{DIR_CFGS.PRETRAINED_DIR}/nuclei_model.pth'
CELL_MODEL = f'{DIR_CFGS.PRETRAINED_DIR}/cell_model.pth'

PI  = np.pi
INF = np.inf
EPS = 1e-12

ID             = 'ID'
LABEL          = 'Label'
WIDTH          = 'ImageWidth'
HEIGHT         = 'ImageHeight'
TARGET         = 'PredictionString'
CELL_LINE      = 'Cellline'
ANTIBODY       = 'Antibody'
ANTIBODY_LABEL = 'AntibodyLabel'

NUM_CLASSES = 19
ML_NUM_CLASSES = 20000

NEGATIVE = 18
LABEL_TO_NAME = {
  0: 'Nucleoplasm',
  1: 'Nuclear membrane',
  2: 'Nucleoli',
  3: 'Nucleoli fibrillar center',
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',
  6: 'Endoplasmic reticulum',
  7: 'Golgi apparatus',
  8: 'Intermediate filaments',
  9: 'Actin filaments',
  10: 'Microtubules',
  11: 'Mitotic spindle',
  12: 'Centrosome',
  13: 'Plasma membrane',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'Vesicles and punctate cytosolic patterns',
  NEGATIVE: 'Negative',
}
LABEL_TO_ALIAS = {
  0: 'Nucleoplasm',
  1: 'NuclearM',
  2: 'Nucleoli',
  3: 'NucleoliFC',
  4: 'NuclearS',
  5: 'NuclearB',
  6: 'EndoplasmicR',
  7: 'GolgiA',
  8: 'IntermediateF',
  9: 'ActinF',
  10: 'Microtubules',
  11: 'MitoticS',
  12: 'Centrosome',
  13: 'PlasmaM',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'VesiclesPCP',
  NEGATIVE: 'Negative',
}
NAMES = [LABEL_TO_NAME[i] for i in range(NUM_CLASSES)]
ALIASES = [LABEL_TO_ALIAS[i] for i in range(NUM_CLASSES)]

COLORS = ['red', 'green', 'blue', 'yellow']