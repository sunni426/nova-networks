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
from sklearn.decomposition import PCA
import umap    
import joblib

osp = os.path
ope = os.path.exists
opj = os.path.join

#%%
FEATURE_DIR    = '/home/trangle/HPA_SingleCellClassification/team1_bestfitting/features'
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


def prepare_train_features():
    valid_df = pd.read_csv('/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/split/random_folds5/random_valid_cv0.csv')
    valid_df.shape

    df = pd.read_csv(f'/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/mask/{DATASET}.csv')
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
    cell_df = pd.read_csv(f'/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/inputs/cellv4b_{DATASET}.csv')
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

    # Load trainval features
    file_name = f'cell_features_{DATASET}_default_cell_v1_trainvalid.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name}'
    features = np.load(features_file, allow_pickle=True)['feats']
    features.shape

    # filter train features
    train_df = df[(df['target']!='unknown') & (df['valid']==0)]
    train_df.target.value_counts()
    #train_df = train_df.groupby('target').head(2000)
    train_df.to_csv(f'/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/inputs/cellv4b_{DATASET}_subsettrain.csv')
    train_features = features[train_df.index]
    train_features.shape
    np.savez_compressed(features_file.replace("trainvalid", "train"), feats=train_features)

def prepare_meta_publicHPA():
    cells_publicHPA_path = f'{DATA_DIR}/inputs/cells_publicHPA.csv'
    if os.path.exists(cells_publicHPA_path):
        print(f'Loading {cells_publicHPA_path}')
        ifimages_v20_ = pd.read_csv(cells_publicHPA_path)
    else:
        ifimages_v20_ = pd.read_csv(f"{DATA_DIR}/inputs/{DATASET}.csv")
        masks = pd.read_csv(f"{DATA_DIR}/mask/{DATASET}.csv")
        ifimages_v20_ = ifimages_v20_.merge(masks, how='inner', on=['ID', 'ImageHeight', 'ImageWidth'])
    
        labels = ifimages_v20_[LABEL_ALIASE_LIST].values
        single_label_idx = np.where((labels==1).sum(axis=1)==1)[0]
        single_labels = labels[single_label_idx]
        idx1 = np.where(single_labels==1)
        single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]
        multi_label_idx = np.where((labels==1).sum(axis=1)>1)[0]
        multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]
    
        ifimages_v20_['target'] = 'unknown'
        ifimages_v20_.loc[single_label_idx, 'target'] = single_labels
        ifimages_v20_.loc[multi_label_idx, 'target'] = multi_labels
    
        ifimages_v20_.target.value_counts()
        #ifimages_v20_.to_csv(f'{DATA_DIR}/inputs/cells_publicHPA.csv', index=False)
    
    ### Merge HPA image-level labels and predicted SC levels from bestfitting best single model
    # Do some rules like bestfitting to combine image-level labels and predicted SC labels?
    prediction = pd.read_csv(f'{FEATURE_DIR.replace("features","result")}/{MODEL_NAME}/fold0/epoch_12.00_ema/cell_result_test_cell_v1.csv')
    prediction["cellmask"] = prediction["mask"]
    prediction = prediction.drop(columns=["mask"])    
    tmp = prediction.merge(ifimages_v20_, how='inner', on=['ID', 'cellmask','maskid'])
    
    il_labels = tmp[[l+'_y' for l in LABEL_ALIASE_LIST]].values
    sc_labels = tmp[[l+'_x' for l in LABEL_ALIASE_LIST]].values
    sc_labels = np.array([c/c.max() for c in sc_labels])
    # sc_labels = list(map(lambda row: [roundToNearest(c, 0.25) for c in row], sc_labels))
    sc_labels = [roundToNearest(c, 0.25) for c in sc_labels]
    sc_labels = pd.DataFrame(np.round(il_labels*sc_labels).astype('uint8'))
    sc_labels.columns = LABEL_ALIASE_LIST
    
    df_c = pd.concat([tmp[["ID", "Label", "maskid", 'cellmask', 'ImageWidth', 'ImageHeight']], sc_labels], axis=1)
    labels = df_c[LABEL_ALIASE_LIST].values
    negatives_idx = np.where(labels.sum(axis=1)==0)[0]
    df_c.loc[negatives_idx, "Negative"] = 1
    
    single_label_idx = np.where((labels==1).sum(axis=1)==1)[0]
    single_labels = labels[single_label_idx]
    idx1 = np.where(single_labels==1)
    single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]
    multi_label_idx = np.where((labels==1).sum(axis=1)>1)[0]
    multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]

    df_c['target'] = 'Negative'
    df_c.loc[single_label_idx, 'target'] = single_labels
    df_c.loc[multi_label_idx, 'target'] = multi_labels

    df_c.target.value_counts()
    df_c.to_csv(f'{DATA_DIR}/inputs/cells_publicHPA_mergedSCprediction_quarterthreshold.csv', index=False)
    
    """
    # Reorganize cells_publicHPA.csv for correct cell order with features file
    labels = il_labels
    il_labels = pd.DataFrame(il_labels)
    il_labels.columns = LABEL_ALIASE_LIST
    # = pd.concat([df_c[["ID", "Label", "maskid", 'cellmask', 'ImageWidth', 'ImageHeight']], il_labels], axis=1)

    single_label_idx = np.where((labels==1).sum(axis=1)==1)[0]
    single_labels = labels[single_label_idx]
    idx1 = np.where(single_labels==1)
    single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]
    multi_label_idx = np.where((labels==1).sum(axis=1)>1)[0]
    multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]

    ifimages_v20_['target'] = 'Negative'
    ifimages_v20_.loc[single_label_idx, 'target'] = single_labels
    ifimages_v20_.loc[multi_label_idx, 'target'] = multi_labels
    ifimages_v20_.target.values_count()
    ifimages_v20_.to_csv(f'{DATA_DIR}/inputs/cells_publicHPA.csv', index=False)
    """
    
def roundToNearest(inputNumber, base=0.25):
    return base*np.round(inputNumber/base)

def fit_umap(preprocess=True):
    if preprocess:
        # Load train features:
        file_name = f'cell_features_{DATASET}_default_cell_v1_train.npz'
        features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name}'
        train_features = np.load(features_file, allow_pickle=True)['feats']
        print("train_features loaded with shape ", train_features.shape)

        # Load publicHPA features
        file_name_publicHPA = 'cell_features_train_default_cell_v1_publicHPA.npz'
        features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name_publicHPA}'
        features_publicHPA = np.load(features_file, allow_pickle=True)['feats']
        print("publicHPA_features loaded with shape ", features_publicHPA.shape)

        # Preprocess
        X = preprocessing.scale(np.vstack((train_features, features_publicHPA)))
        train_features = X[:len(train_features)]

        file_name = f'cell_features_{DATASET}_default_cell_v1_train_preprocessed.npz'
        features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name}'
        np.savez_compressed(features_file, feats=train_features)
        print("train_features processed, new shape ", train_features.shape)
        
        features_publicHPA = X[len(train_features):]
        file_name_publicHPA = 'cell_features_train_default_cell_v1_publicHPA_preprocessed.npz'
        features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name_publicHPA}'
        np.savez_compressed(features_file, feats=features_publicHPA)
        print("features_publicHPA processed, new shape ", train_features.shape)
    else:
        # Load train features:
        file_name = f'cell_features_{DATASET}_default_cell_v1_train_preprocessed.npz'
        features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name}'
        train_features = np.load(features_file, allow_pickle=True)['feats']
        print("processed train_features loaded with shape ", train_features.shape)

        #file_name_publicHPA = 'cell_features_train_default_cell_v1_publicHPA_preprocessed.npz'
        #features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name_publicHPA}'
        #features_publicHPA = np.load(features_file, allow_pickle=True)['feats']
        #print("processed publicHPA_features loaded with shape ", features_publicHPA.shape)

    # Fit (and transform)
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        metric='euclidean', 
        random_state=33,
        n_epochs =100,
        transform_queue_size=2, 
        low_memory = False,
        verbose=True)
    reducer.fit(train_features)
    print("umap fitted to train_features")

    filename = f"{DATA_DIR}/fitted_umap.sav"
    joblib.dump(reducer, filename)

    #X = reducer.transform(features_publicHPA.tolist())
    #np.savez_compressed(f"{DATA_DIR}/transformed_publicHPA.npz", feats=X)

def transform_umap():    
    # Load publicHPA features
    file_name_publicHPA = 'cell_features_train_default_cell_v1_publicHPA_preprocessed.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name_publicHPA}'
    features_publicHPA = np.load(features_file, allow_pickle=True)['feats'][:500000]
    print("transformed publicHPA_features loaded with shape ", features_publicHPA.shape)

    # Fit (and transform)
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        metric='euclidean', 
        random_state=20,
        n_epochs =100,
        transform_queue_size=2, 
        low_memory = False,
        verbose=True)
    X = reducer.fit_transform(features_publicHPA.tolist())
    np.savez_compressed(f"{DATA_DIR}/transformed_publicHPA.npz", feats=X)
    

def transform_points(reducer, points):
    # Super slow, time increases astronomically 
    transformed_X = []
    for i in range(len(points)):
        transformed_X += [reducer.transform(points[i: i+1].tolist())]
    return np.array(transformed_X)

def plot_umap(show_multi=True,title=''):
    X = np.load(f"{DATA_DIR}/transformed_publicHPA.npz", allow_pickle=True)['feats']
    sub_df = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA.csv')[:50000]
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
    plt.savefig(f"{DATA_DIR}/{title}.png")

    sub_df["x"] = [idx[0] for idx in X]
    sub_df["y"] = [idx[1] for idx in X]
    sub_df["id"] = ["_".join([img,str(cell)]) for img, cell in zip(sub_df.ID, sub_df.maskid)]
    sub_df.to_csv(f"{DATA_DIR}/{title}.csv", index=False)

def show_features_separate(train_features, features_publicHPA, sub_df, show_multi=True, title=''):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',random_state=33, verbose=True)
    reducer.fit(train_features)
    X = reducer.transform(features_publicHPA.tolist())

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
    sub_df.to_csv(f"{DATA_DIR}/{title}.csv", index=False)
    return sub_df


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
        verbose=True)
    X = reducer.fit_transform(features.tolist())

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
    plt.savefig(f"/home/trangle/HPA_SingleCellClassification/plots/umap/{title}.png")
    
    sub_df["x"] = [idx[0] for idx in X]
    sub_df["y"] = [idx[1] for idx in X]
    if umap_args['n_components'] == 3:
        sub_df["z"] = [idx[2] for idx in X]
    sub_df["id"] = ["_".join([img,str(cell)]) for img, cell in zip(sub_df.ID, sub_df.maskid)]
    sub_df.to_csv(f"{DATA_DIR}/{title}.csv", index=False)
    return sub_df
#%%
def main():
    #prepare_train_features()
    #prepare_meta_publicHPA()
    #fit_umap()
    #fit_transform_plot()
    #transform_umap()
    #plot_umap(show_multi=True,title='publicHPA_multilocalization')
    
    '''
    # Load train features:
    DATASET='train'
    file_name = f'cell_features_{DATASET}_default_cell_v1_train.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name}'
    train_features = np.load(features_file, allow_pickle=True)['feats']
    print("train_features loaded with shape ", train_features.shape)
    train_df = pd.read_csv(f'/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/inputs/cellv4b_{DATASET}_subsettrain.csv')
    '''

    # Load publicHPA features
    file_name_publicHPA = 'cell_features_test_default_cell_v1.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name_publicHPA}'
    features_publicHPA_0 = np.load(features_file, allow_pickle=True)['feats']
    print("publicHPA_features loaded with shape ", features_publicHPA_0.shape)
    public_hpa_df = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA_mergedSCprediction.csv')
    public_hpa_df0 = pd.read_csv(f'{DATA_DIR}/inputs/cells_publicHPA.csv')
    
    #label_df = train_df.groupby('target').head(3000)
    #feature_matrix = train_features[label_df.index]
    
    label_df = public_hpa_df.groupby('target').head(10000)
    feature_matrix =  features_publicHPA_0[label_df.index]  
    args = dict({
        'n_neighbors':15, 
        'min_dist':0.1, 
        'n_components':2, 
        'metric':'euclidean'
    })
    plottile = f'ml_pHPA10000_15_0.1_euclidean_2d_il*sc'
    show_features_fit_transform(feature_matrix, label_df, args, pca=False, show_multi=True, title=plottile)  
    plottile = f'sl_pHPA10000_15_0.1_euclidean_2d_il*sc'
    show_features_fit_transform(feature_matrix, label_df, args, pca=False, show_multi=False, title=plottile)  

    label_df = public_hpa_df.groupby('target').head(10000)
    feature_matrix =  features_publicHPA_0[label_df.index]    
    
    """ Performing PCA
    pca = PCA(n_components=10)
    feature_matrix = preprocessing.scale(feature_matrix)
    feature_matrix = pca.fit_transform(feature_matrix)
    pca.explained_variance_ratio_

    lovain
    scanpy
    testg pca with small data
    """
    for n in [10,20]:
        for d in [0.1,0.25,0.5]:
            for m in ['euclidean', 'braycurtis']:
                args = dict({
                    'n_neighbors':n, 
                    'min_dist':d, 
                    'n_components':2, 
                    'metric':m
                })
                plottile = f'sl_pHPA10000_{n}_{d}_{m}_2d'
                show_features_fit_transform(feature_matrix, label_df, args, pca=False, show_multi=False, title=plottile)
    
    import gc
    gc.collect()
    """
    label_df = public_hpa_df.groupby('target').head(100000)
    feature_matrix =  features_publicHPA_0[label_df.index]     
    for n in [15,20]:
        for d in [0.1,0.5]:
            for m in ['euclidean']:
                args = dict({
                    'n_neighbors':n, 
                    'min_dist':d, 
                    'n_components':2, 
                    'metric':m
                })
    plottile = f'sl_pHPA100000_pca_{n}_{d}_{m}_2d'
    show_features_fit_transform(feature_matrix, label_df, args, pca=True, show_multi=False, title=plottile)
    """
    import re
    ### Prepare data for web app
    ifimages = pd.read_csv("/data/kaggle-dataset/PUBLICHPA/raw/train.csv")
    ifimages_v20 = ifimages[ifimages.latest_version == 20.0]
    ifimages_v20 = add_label_idx(ifimages_v20, all_locations)
    ifimages_v20["ID"] = [os.path.basename(f)[:-1] for f in ifimages_v20.filename]
    ifimages_v20.to_csv("/data/kaggle-dataset/publicHPA_umap/ifimages_v20.csv", index=False)
    result_df = pd.read_csv(f"{DATA_DIR}/sl_pHPA10000_15_0.1_braycurtis_pca100_3d.csv")
    
    tmp = result_df.merge(ifimages_v20, how='left', on=['ID', 'Label'])
    tmp["Ab"] = [str(int(re.findall("\d+", t)[0])) for t in tmp.antibody]
    tmp["id"] = tmp["Ab"] + "_" + tmp["id"]
    
    LABEL_ALIASE_LIST = [LABEL_TO_ALIAS[i] for i in range(NUM_CLASSES)]
    tmp["location_code"] = [LABEL_ALIASE_LIST.index(l) for l in tmp.target]
    tmp = tmp[["x","y","id","location_code","locations","gene_names","ensembl_ids","atlas_name","cellmask","ImageWidth", "ImageHeight","target"]]
    tmp.to_csv(f"{DATA_DIR}/sl_pHPA10000_15_0.1_braycurtis_pca100_3d_webapp.csv", index=False)    
    tmp[:100000].to_csv(f"{DATA_DIR}/sl_pHPA10000_15_0.1_braycurtis_pca100_3d_webapp_100k.csv", index=False)    
    
    
def get_numbers(string):
    num = [int(s) for s in string if s.isdigit()]
    return 

if __name__ == '__main__':
    main()
