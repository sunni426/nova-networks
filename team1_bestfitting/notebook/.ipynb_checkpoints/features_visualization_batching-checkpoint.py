#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:20:58 2022

@author: trangle
"""
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA


FEATURE_DIR    = '/home/trangle/HPA_SingleCellClassification/team1_bestfitting/features'
DATA_DIR       = '/data/kaggle-dataset/PUBLICHPA'
DATASET        = 'train'
MODEL_NAME     = 'd0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds'
 
NUM_CLASSES = 20
NEGATIVE = 18
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
COLORS = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#dddddd',
    '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
    '#00e676', '#64ffda', '#18ffff',
]

def show_features_fit_transform(features, sub_df, umap_args, pca=False, show_multi=True, title=''):
    if pca:
        pca_pp = PCA(n_components=100)
        features = preprocessing.scale(features)
        features = pca_pp.fit_transform(features)
        print(f'Percent of explained variance: {np.cumsum(pca_pp.explained_variance_ratio_)[-1]*100}%')
        plt.bar(range(1,len(pca_pp.explained_variance_ )+1),pca_pp.explained_variance_ )
        plt.ylabel('Explained variance')
        plt.xlabel('Components')
        plt.plot(range(1,len(pca_pp.explained_variance_ )+1),
                 np.cumsum(pca_pp.explained_variance_),
                 c='red',
                 label="Cumulative Explained Variance")
        plt.legend(loc='upper left')
    else:
        features = preprocessing.scale(features, axis=0)
    reducer = umap.UMAP(
        n_neighbors=umap_args['n_neighbors'], 
        min_dist=umap_args['min_dist'], 
        n_components=umap_args['n_components'], 
        metric=umap_args['metric'],
        random_state=99, 
        n_jobs=10,
        verbose=True)
    X = reducer.fit_transform(features.tolist())
    
    if False:
        rm_outliers = abs(X[:,0])<6
        print(len(rm_outliers), rm_outliers)
        X = X[rm_outliers,:]
        sub_df = sub_df[rm_outliers]

    num_classes = NUM_CLASSES if show_multi else NUM_CLASSES-1
    fig, ax = plt.subplots(figsize=(32, 16))
    for i in range(num_classes):
        label = LABEL_TO_ALIAS[i]
        if label in ['Negative', 'Multi-Location']:
            continue
        idx = np.where(sub_df['target']==label)[0]
        x = X[idx, 0]
        y = X[idx, 1]
        #print(label, sub_df['Label'][idx])
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=16, alpha=0.5)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1, markerscale=6)
    plt.title(title, fontsize=24)
    plt.savefig(f"/home/trangle/HPA_SingleCellClassification/plots/umap/{title}.png")
    
    sub_df["x"] = [idx[0] for idx in X]
    sub_df["y"] = [idx[1] for idx in X]
    if umap_args['n_components'] == 3:
        sub_df["z"] = [idx[2] for idx in X]
    sub_df["id"] = ["_".join([img,str(cell)]) for img, cell in zip(sub_df.ID, sub_df.maskid)]
    sub_df.to_csv(f"{DATA_DIR}/{title}.csv", index=False)
    return sub_df
    

def main():
    #df = pd.read_csv("./Downloads/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d.csv")
    
    # Load train features:
    file_name = 'cell_features_test_default_cell_v1.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name}'
    train_features = np.load(features_file, allow_pickle=True)['feats']
    print("processed train_features loaded with shape ", train_features.shape)
    public_hpa_df = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA_mergedSCprediction.csv')
    public_hpa_df0 = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA.csv')
    
    umap_args = dict({
        'n_neighbors':15, 
        'min_dist':0.1, 
        'n_components':2,
        'n_epochs':300,
        'metric':'euclidean'
    })
    
    for i in range(5):
        sub_df = public_hpa_df0.sample(frac=1, random_state=i).head(300000)#.groupby('target').head(20000)
        features =  train_features[sub_df.index]  
        print(features.shape)
        show_features_fit_transform(features, sub_df, umap_args, pca=False, show_multi=True, title=f'randombatch_{i}_nogrouping')
            
if __name__ == '__main__':
    main()

