import os
import re
import pandas as pd
import base64
import zlib
from pycocotools import _mask as coco_mask
from skimage.measure import regionprops

FEATURE_DIR    = './features'
DATA_DIR       = './PUBLICHPA'
DATASET        = 'train'
MODEL_NAME     = 'd0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds'
SAVE_DIR       = './results'
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
NUM_CLASSES = 20
LABEL_ALIASE_LIST = [LABEL_TO_ALIAS[i] for i in range(NUM_CLASSES-1)]

def add_label_idx(df, all_locations):
    '''Function to convert label name to index
    '''
    df["Label"] = None
    for i, row in df.iterrows():
        locations = str(row.locations)
        if locations == 'nan':
            continue
        labels = locations.split(',')
        idx = []
        for l in labels:
            if l in all_locations.keys():
                idx.append(str(all_locations[l]))
        if len(idx)>0:
            df.loc[i,"Label"] = "|".join(idx)
            
        print(df.loc[i,"locations"], df.loc[i,"Label"])
    return df

def decodeToBinaryMask(rleCodedStr, imWidth, imHeight):
    # s = time.time()
    uncodedStr = base64.b64decode(rleCodedStr)
    uncompressedStr = zlib.decompress(uncodedStr, wbits=zlib.MAX_WBITS)
    detection = {"size": [imWidth, imHeight], "counts": uncompressedStr}
    detlist = []
    detlist.append(detection)
    mask = coco_mask.decode(detlist)
    binaryMask = mask.astype("uint8")[:, :, 0]
    # print(f'Decoding 1 cell: {time.time() - s} sec') #Avg 0.0035 sec for each cell
    return binaryMask 

def main():
    ### Prepare data for web app
    filename = 'pHPA10000_15_0.1_euclidean_ilsc_2d'
    result_df = pd.read_csv(f"{SAVE_DIR}/umap/{filename}.csv")
    ifimages_v20 = pd.read_csv(f"{DATA_DIR}/ifimages_v20.csv")
    
    tmp = result_df.merge(ifimages_v20, how='left', on=['ID', 'Label'])
    tmp["Ab"] = [str(int(re.findall("\d+", t)[0])) for t in tmp.antibody]
    tmp["id"] = tmp["Ab"] + "_" + tmp["id"]

    tmp["location_code"] = ""
    tmp["location"] = ""
    for i, row in tmp.iterrows():
        loc = []
        for col in LABEL_ALIASE_LIST:
            if row[col] == 1:
                loc += [col]
        #print(loc)
        if len(loc) == 1:
            row.location = loc[0]
            row.location_code = LABEL_ALIASE_LIST.index(loc[0])
        else:
            row.location = ",".join(loc)
            row.location_code = " ".join([str(LABEL_ALIASE_LIST.index(l)) for l in loc])

    #LABEL_ALIASE_LIST = [LABEL_TO_ALIAS[i] for i in range(NUM_CLASSES)]
    #tmp["location_code"] = [LABEL_ALIASE_LIST.index(l) for l in tmp.target]
    try:
        tmp = tmp[["x","y","z","id","location","location_code","locations","gene_names","ensembl_ids","atlas_name","cellmask","ImageWidth", "ImageHeight","target"]]
    except:
        tmp = tmp[["x","y","id","location","location_code","locations","gene_names","ensembl_ids","atlas_name","cellmask","ImageWidth", "ImageHeight","target"]]

    if not os.path.isdir(f"{SAVE_DIR}/webapp"):
        os.mkdir(f"{SAVE_DIR}/webapp")
    tmp.to_csv(f"{SAVE_DIR}/webapp/{filename}.csv", index=False)    
    
    # (Cell center X, center Y + height and width) or boundingbox size (upper left, lower right)
    tmp["top"] = 0
    tmp["left"] = 0
    tmp["width"] = 0
    tmp["height"] = 0
    for i, row in tmp.iterrows():
        mask1 = decodeToBinaryMask(row.cellmask, row.ImageWidth, row.ImageHeight)
        region = regionprops(mask1)[0]
        minr, minc, maxr, maxc = region.bbox
        tmp.loc[i,"top"] = minr #/row.ImageWidth
        tmp.loc[i,"left"] = minc # /row.ImageHeight
        tmp.loc[i,"width"] = maxr - minr
        tmp.loc[i,"height"] = maxc - minc
    tmp.drop(columns=["cellmask","ImageWidth", "ImageHeight"])
    tmp.to_csv(f"{SAVE_DIR}/webapp/{filename}_bbox.csv", index=False)    

if __name__ == '__main__':
    main()
