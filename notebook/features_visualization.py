import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from numpy.matrixlib.defmatrix import matrix
import numpy as np
import os
from sklearn.metrics import f1_score
#import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
#%matplotlib inline 
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
sns.set(style='white', color_codes=True)


from sklearn import preprocessing
from sklearn.manifold import TSNE
import umap

osp = os.path
ope = os.path.exists
opj = os.path.join

#%%
FEATURE_DIR    = './team1_bestfitting/features'
DATA_DIR       = './team1_bestfitting/data'

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

#%%
DATASET = 'train'
valid_df = pd.read_csv(f'{DATA_DIR}/split/random_folds5/random_valid_cv0.csv')
valid_df.shape

df = pd.read_csv(f'{DATA_DIR}/mask/{DATASET}.csv')
df['valid'] = 0
df.loc[df[ID].isin(valid_df[ID].values), 'valid'] = 1
print(df.shape)
df.head()

# Here is the label of the cells, you can use your truth cells label.
# The cells were labeled to 5 levels with label [1.0, 0.75, 0.5, 0.25, 0 ], 
# this is a rule based procedure, After getting the outputs of all cells of train set from FCAN, 
# we can give higher label value if the image probability and cell probability are high, 
# and the cells from an image with label A were given at least 0.25 of this label A. 
# The thresholds of the rule were not sensitive according to my experiments.
cell_df = pd.read_csv(f'{DATA_DIR}/inputs/cellv4b_{DATASET}.csv')
print(cell_df.shape)
cell_df.head()

df = df.merge(cell_df, how='left', on=[ID, 'maskid'])
df.shape

labels = df[LABEL_ALIASE_LIST].values
single_label_idx = np.where((labels==1).sum(axis=1)==1)[0]
single_labels = labels[single_label_idx]
idx1 = np.where(single_labels==1)
single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]

multi_label_idx = np.where((labels==1).sum(axis=1)>1)[0]
multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]

df['target'] = 'unknown'
df.loc[single_label_idx, 'target'] = single_labels
df.loc[multi_label_idx, 'target'] = multi_labels
df['target'].value_counts()


#%%
model_name = 'd0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds'
file_name = f'cell_features_{DATASET}_default_cell_v1_trainvalid.npz'
features_file = f'{FEATURE_DIR}/{model_name}/fold0/epoch_12.00_ema/{file_name}'
features = np.load(features_file, allow_pickle=True)['feats']
features.shape

train_df = df[(df['target']!='unknown') & df['valid']==0]
train_df = train_df.groupby('target').head(1000)
train_features = features[train_df.index]
train_features.shape

valid_df = df[(df['target']!='unknown') & df['valid']==1]
valid_df = valid_df.groupby('target').head(1000)
valid_features = features[valid_df.index]
valid_features.shape

X = preprocessing.scale(np.vstack((train_features, valid_features)))
train_features = X[:len(train_features)]
valid_features = X[len(train_features):]
print(train_features.shape,valid_features.shape)


def show_features(predict_valid=True, show_multi=True, title=''):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',random_state=33)
    if predict_valid:
        reducer.fit(train_features)
        X = reducer.transform(valid_features.tolist())
        sub_df = valid_df
    else:
        X = reducer.fit_transform(train_features)
        sub_df = train_df

    num_classes = NUM_CLASSES if show_multi else NUM_CLASSES-1
    fig, ax = plt.subplots(figsize=(32, 16))
    for i in range(num_classes):
        label = LABEL_TO_ALIAS[i]
        idx = np.where(sub_df['target']==label)[0]
        x = X[idx, 0]
        y = X[idx, 1]
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1)
    plt.title(title, fontsize=24)
    sub_df["x"] = [idx[0] for idx in X]
    sub_df["y"] = [idx[1] for idx in X]
    sub_df["id"] = ["_".join([img,str(cell)]) for img, cell in zip(sub_df.ID, sub_df.maskid)]
    sub_df.to_csv(f"{DATA_DIR}/{title}.csv", index=False) #/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/multi-location.csv and single-location.csv
    return sub_df
    
#%%
title = 'multi-location'
sub_df = show_features(predict_valid=False, show_multi=True, title=title)

target_format = pd.read_csv("/home/trangle/Downloads/umap_results_fit_all_transform_all_sorted_20190422.csv")

predict_valid = True
show_multi = True
if True:
    
    title = 'valid_onefifth_concat'
    parts = len(valid_features)//5
    X = []
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',random_state=33)
    reducer.fit(train_features)
    X += [reducer.transform(valid_features[:parts].tolist())]
    X += [reducer.transform(valid_features[parts:2*parts].tolist())]
    X += [reducer.transform(valid_features[2*parts:3*parts].tolist())]
    X += [reducer.transform(valid_features[3*parts:4*parts].tolist())]
    X += [reducer.transform(valid_features[4*parts:].tolist())]
    #X = reducer.transform(valid_features.tolist())
    sub_df = valid_df
    
    X = np.concatenate(X)

    """
    title = 'train'
    X = reducer.fit_transform(train_features)
    sub_df = train_df
    """
    num_classes = NUM_CLASSES if show_multi else NUM_CLASSES-1
    fig, ax = plt.subplots(figsize=(32, 16))
    for i in range(num_classes):
        label = LABEL_TO_ALIAS[i]
        idx = np.where(sub_df['target']==label)[0]
        x = X[idx, 0]
        y = X[idx, 1]
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1)
    plt.title(title, fontsize=24)

#%% Shuffle valid indexes
import random
shuffle = True
split_n_parts = False
if True:
    
    title = 'valid_shuffle_all'
    if shuffle:
        valid_df_reindex = valid_df.reset_index() # reset indexes to 0,1,2,...
        random.seed(10)
        indexes = list(range(len(valid_df_reindex)))
        random.shuffle(indexes)
        #sub_df = valid_df.sample(frac = 1, random_state=10)
        #order = list(sub_df.index)
        
        sub_df = valid_df_reindex.iloc[indexes]
        valid_features_ordered = valid_features[indexes,:]
    else:
        sub_df = valid_df
        valid_features_ordered = valid_features
    
    if split_n_parts:
        parts = len(valid_features)//5
        X = []
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',random_state=33)
        reducer.fit(train_features)
        X += [reducer.transform(valid_features_ordered[:parts].tolist())]
        X += [reducer.transform(valid_features_ordered[parts:2*parts].tolist())]
        X += [reducer.transform(valid_features_ordered[2*parts:3*parts].tolist())]
        X += [reducer.transform(valid_features_ordered[3*parts:4*parts].tolist())]
        X += [reducer.transform(valid_features_ordered[4*parts:].tolist())]
        X = np.concatenate(X)
    else:
        X = reducer.transform(valid_features_ordered.tolist())
        
    #X = reducer.fit_transform(valid_features_ordered[4*parts:].tolist())
    #sub_df = valid_df[4*parts:]
    num_classes = NUM_CLASSES if show_multi else NUM_CLASSES-1
    fig, ax = plt.subplots(figsize=(32, 16))
    for i in range(num_classes):
        label = LABEL_TO_ALIAS[i]
        idx = np.where(sub_df['target']==label)[0]
        x = X[idx, 0]
        y = X[idx, 1]
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1)
    plt.title(title, fontsize=24)

#%% Gradually reduce part size

def plot_umap(train_features, valid_features, sub_df, title=''):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', verbose=True)
    reducer.fit(train_features)
    
    X = reducer.transform(valid_features.tolist())
    
    num_classes = NUM_CLASSES if show_multi else NUM_CLASSES-1
    fig, ax = plt.subplots(figsize=(32, 16))
    for i in range(num_classes):
        label = LABEL_TO_ALIAS[i]
        idx = np.where(sub_df['target']==label)[0]
        x = X[idx, 0]
        y = X[idx, 1]
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1)
    plt.title(title, fontsize=24)

#%%
valid_df_ori = valid_df.copy()


valid_df = valid_df_ori.reset_index() # reset indexes to 0,1,2,...
random.seed(10)
indexes = list(range(len(valid_df)))
random.shuffle(indexes)
sub_df_ordered = valid_df.iloc[indexes]
valid_features_ordered = valid_features[indexes,:]


title = 'valid_shuffle_80%'
n = int(len(valid_df)*0.8)
plot_umap(train_features, valid_features_ordered[:n], sub_df_ordered[:n], title)
