from pathlib import Path
import argparse
from skimage.io import imread, imsave
import numpy as np
from cellpose import models, io
import multiprocessing as mp
from tqdm import tqdm
import torch
import pandas as pd

def merge_channels(base_dir, image_id):
    # Construct filepaths for red, green, and blue images
    red_path = base_dir.joinpath(f"{image_id}_red_merged.png")
    green_path = base_dir.joinpath(f"{image_id}_green.png")
    blue_path = base_dir.joinpath(f"{image_id}_blue.png")

    # Read the images
    red_image = imread(red_path)
    green_image = imread(green_path)
    blue_image = imread(blue_path)

    # Stack the images to form an RGB image
    merged_image = np.stack([red_image, green_image, blue_image], axis=-1)
    return merged_image

def cellpose_seg(args):
    # Read the CSV file and extract the IDs
    csv_path = Path(args.i).joinpath('0_train_240220.csv')
    df = pd.read_csv(csv_path)
    image_ids = df['ID'].tolist()

    # Base directory
    base_dir = Path('/project/btec-design3/kaggle_dataset/toy_dataset')
    opdir = Path(args.o)
    opdir.mkdir(parents=True, exist_ok=True)

    model = models.Cellpose(gpu=True, model_type='cyto2', device=torch.device(f'cuda:{args.gpu_device}'))

    # Process images in batches
    for image_id in tqdm(image_ids):
        merged_image = merge_channels(base_dir, image_id)
        masks, flows, styles, diams = model.eval([merged_image], 
                                                 diameter=150, 
                                                 channels=[2, 3], 
                                                 flow_threshold=0.8,
                                                 cellprob_threshold=0.0,
                                                 batch_size=args.batch_size)
        
        opdatapath = opdir.joinpath(f'{image_id}.npy')
        opsegpath = opdir.joinpath(f'{image_id}.png')
        io.masks_flows_to_seg([merged_image], masks, flows, diams, [opdatapath])
        io.save_to_png([merged_image], masks, flows, [opsegpath])
    
    return

def main():
    description = "Cellpose segmentation script with channel merging"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', type=str, required=True, help='input dir with image_list.csv')
    parser.add_argument('-o', type=str, required=True, help='output dir')
    parser.add_argument('-g', '--gpu_device', type=int, default=0, help='gpu device')    
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='batch size')
    args = parser.parse_args()
    cellpose_seg(args) 
    return    

if __name__ == "__main__":
    main()