#import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
import umap    

FEATURE_DIR    = '/home/trangle/HPA_SingleCellClassification/hpa2021_0902/features'
DATA_DIR       = '/data/kaggle-dataset/PUBLICHPA'
DATASET        = 'train'
MODEL_NAME     = 'd0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds'

ID             = 'ID'
LABEL          = 'Label'
WIDTH          = 'ImageWidth'
HEIGHT         = 'ImageHeight'
TARGET         = 'PredictionString'
CELL_LINE      = 'Cellline'
ANTIBODY       = 'Antibody'
ANTIBODY_LABEL = 'AntibodyLabel'

NUM_CLASSES = 20
ML_NUM_CLASSES = 20000

NEGATIVE = 18

LABEL_NAMES = {
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
  19:'Multi-Location',
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
  19:'Multi-Location',
}
LABEL_NAME_LIST = [LABEL_NAMES[i] for i in range(NUM_CLASSES-1)]
LABEL_ALIASE_LIST = [LABEL_TO_ALIAS[i] for i in range(NUM_CLASSES-1)]

COLORS = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#dddddd',
    '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
    '#00e676', '#64ffda', '#18ffff',
]


def main():
    # Load publicHPA features
    file_name_publicHPA = 'cell_features_test_default_cell_v1.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name_publicHPA}'
    features_publicHPA_0 = np.load(features_file, allow_pickle=True)['feats']
    print("publicHPA_features loaded with shape ", features_publicHPA_0.shape)
    #public_hpa_df = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA_mergedSCprediction.csv')
    public_hpa_df0 = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA.csv')

    # PCA
    pca = PCA(n_components=70)
    #features = preprocessing.scale(features_publicHPA_0)
    features = features_publicHPA_0
    features = pca.fit_transform(features)
    print(f'Explained variance after PCA: {sum(pca.explained_variance_ratio_)}')

    # UMAP
    umap_args = dict({
        'n_neighbors':15, 
        'min_dist':0.1, 
        'n_components':3, 
        'metric':'braycurtis'
    })
    reducer = umap.UMAP(
        n_neighbors=umap_args['n_neighbors'], 
        min_dist=umap_args['min_dist'], 
        n_components=umap_args['n_components'], 
        metric=umap_args['metric'],
        random_state=33, 
        verbose=True)
    X = reducer.fit_transform(features.tolist())

    
    #public_hpa_df = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA_mergedSCprediction.csv')
    sub_df = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA.csv')
    # Plot 1
title = f'sl_pHPA10000_{15}_{0.1}_braycurtis_pca100_3d'
show_multi = False
num_classes = NUM_CLASSES if show_multi else NUM_CLASSES-1
fig, ax = plt.subplots(figsize=(32, 16))
for i in range(num_classes):
    label = LABEL_TO_ALIAS[i]
    idx = np.where(sub_df['target']==label)[0]
    x = X[idx, 0]
    y = X[idx, 1]
    #print(label, sub_df['Label'][idx])
    plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=16)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1, markerscale=6)
plt.title(title, fontsize=24)
plt.savefig(f"/home/trangle/HPA_SingleCellClassification/plots/umap/{title}_view1.png")

if umap_args['n_components'] == 3:
    # Plot 2
    fig, ax = plt.subplots(figsize=(32, 16))
    for i in range(num_classes):
        label = LABEL_TO_ALIAS[i]
        idx = np.where(sub_df['target']==label)[0]
        x = X[idx, 1]
        y = X[idx, 2]
        #print(label, sub_df['Label'][idx])
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=16)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1, markerscale=6)
    plt.title(title, fontsize=24)
    plt.savefig(f"/home/trangle/HPA_SingleCellClassification/plots/umap/{title}_view2.png")

# Save UMAP coordinates
sub_df["x"] = [idx[0] for idx in X]
sub_df["y"] = [idx[1] for idx in X]
if umap_args['n_components'] == 3:
    sub_df["z"] = [idx[2] for idx in X]
sub_df["id"] = ["_".join([img,str(cell)]) for img, cell in zip(sub_df.ID, sub_df.maskid)]
sub_df.to_csv(f"{DATA_DIR}/{title}.csv", index=False)

if __name__ == '__main__':
    main()