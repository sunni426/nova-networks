import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.exposure import adjust_log, equalize_adapthist
from skimage import img_as_float, img_as_ubyte, img_as_uint
import argparse
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import logging
import glob
import os


def setup_logging(opdir):
    log_file_path = Path(opdir, 'merge_channels.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def _merge_ch(id, ipdir, opdir, chcolors, opframesize, merge03):
    plate_number, well_number, fov = id
    try:
        img = []
        for chcolor in chcolors:
            file_pattern = f"{plate_number}-{chcolor}/cdp2*_{well_number}_{fov}_*.tif"
            file_path = glob.glob(os.path.join(ipdir, file_pattern))
            if file_path:
                img_tmp = imread(file_path[0])
                img.append(img_tmp)
            else:
                raise FileNotFoundError(f"No file found for: {file_pattern}")
       
        img = np.stack(img, axis=-1)
        img = img.astype(np.uint16)
        logging.info(f"Stacked image shape: {img.shape}")
       
        if merge03:
            # Merge the first & the last channel
            if img.shape[-1] > 3:
                img[..., 0] = img[..., 0] + img[..., 3]
                img[..., 0] = np.clip(img[..., 0], 0, 65535)
                img = img[..., :3]
            logging.info(f"Merged image shape: {img.shape}")
        else:
            if img.shape[-1] != 4:
                raise ValueError(f"{file_pattern} doesn't have 4 channels as expected")

        # Adjust image intensity
        for i in range(img.shape[-1]):
            img = img_as_float(img)
            img[..., i] = equalize_adapthist(img[..., i]) 
        logging.info(f"Equalized image shape: {img.shape}")

        # Resize image
        img = resize(img, (opframesize[0], opframesize[1], img.shape[-1]), anti_aliasing=True)
        logging.info(f"Resized image shape: {img.shape}")

        # Save as 8-bit PNG files
        img = img_as_ubyte(img) 
        logging.info(f"Converted image shape: {img.shape}")
        opimgpath = opdir.joinpath(f"{plate_number}_{well_number}_{fov}.png")
        imsave(opimgpath, img)
        logging.info(f"Saved image shape: {img.shape}")
        logging.info(f"Processed image {plate_number}_{well_number}_{fov} successfully")
    except Exception as e:
        logging.error(f"Error processing image {plate_number}_{well_number}_{fov}: {e}")
        return False
    return True


def merge_channels(args):
    ipdir = Path(args.ipdir)
    ipcsvpath = Path(args.ipcsv)
    chcolors = args.channl_order.split(" ")
    opframesize = args.frame_size
    merge_ch0and3 = args.merge_ch0and3

    if len(opframesize) == 1:
        opframesize = tuple([opframesize[0], opframesize[0]])
    elif len(opframesize) == 2:
        opframesize = tuple(opframesize)
    else:
        raise ValueError("frame size should be a tuple of 2 integers")
    opdir = Path(args.opdir)
    opdir.mkdir(exist_ok=True, parents=True)
    n_CPU = args.n_CPU

    id_df = pd.read_csv(ipcsvpath)
    id_list = []
    for _, row in id_df.iterrows():
        plate_number = row['Metadata_Plate']
        well_number = row['Metadata_Well']
        for fov in ['s1', 's2', 's3', 's4', 's5', 's6']:
            id_list.append((plate_number, well_number, fov))

    _merge_channels = partial(
        _merge_ch,
        ipdir=ipdir,
        chcolors=chcolors,
        opdir=opdir,
        opframesize=opframesize,
        merge03=merge_ch0and3,
    )

    if n_CPU != 1:
        pbar = tqdm(total=len(id_list))
        pool = mp.Pool(n_CPU)

        for idx, id in enumerate(id_list):
            pool.apply_async(_merge_channels, (id,), callback=lambda _: pbar.update())

        pool.close()
        pool.join()
        pbar.close()
    else:
        for idx, id in tqdm(enumerate(id_list)):
            success = _merge_channels(id)
            if not success:
                logging.error(f"Failed to process image {id}")

    return


def main():
    description = "Merge channels into a single RGB file"
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group()
    required.add_argument("-i", "--ipdir", type=str, required=True, help="input dir")
    required.add_argument("-ic", "--ipcsv", type=str, required=True, help="input csv")
    required.add_argument("-o", "--opdir", type=str, required=True, help="output img dir")
    optional = parser.add_argument_group()
    optional.add_argument("-ch", "--channl_order", type=str, default="ERSyto Mito Hoechst ERSytoBleed", help="channel order")
    optional.add_argument("-s", "--frame_size", type=int, default=1024, nargs="+", help="frame size for output image")
    optional.add_argument("-n", "--n_CPU", type=int, default=1, help="number of CPU cores to use")
    optional.add_argument("-m", "--merge_ch0and3", action="store_true")
    args = parser.parse_args()
    setup_logging(args.opdir)
    merge_channels(args)
    return

if __name__ == "__main__":
    main()
