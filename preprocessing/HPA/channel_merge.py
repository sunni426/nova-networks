import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.exposure import adjust_log
import argparse
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import logging

def setup_logging(opdir):
    log_file_path = Path(opdir, 'merge_channels.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def _merge_ch(id, ipdir, opdir, chcolors, opframesize, merge03):
    try:
        img = []
        for chcolor in chcolors:
            img_tmp = imread(ipdir.joinpath(f"{id}_{chcolor}.png"))
            print(f"Channel {chcolor}: {img_tmp.shape}")
            img.append(img_tmp)
        img = np.stack(img, axis=-1)
        print(f"Stacked image shape: {img.shape}")

        if merge03:
            # merge yellow and red
            if img.shape[-1] > 3:
                img[..., 0] = img[..., 0] + img[..., 3]
                img[..., 0] = np.clip(img[..., 0], 0, 255)
                img = img[..., :3]
            print(f"Merged image shape: {img.shape}")
        else:
            if img.shape[-1] != 4:
                raise ValueError(f"{id} doesn't have 4 channels as expected")

        img = adjust_log(img, 1)
        img = resize(img, (opframesize[0], opframesize[1], img.shape[2]), anti_aliasing=True)
        img = img * 255
        img = img.astype(np.uint8)
        opimgpath = opdir.joinpath(f"{id}.png")
        imsave(opimgpath, img)
        logging.info(f"Processed image {id} successfully")
    except Exception as e:
        logging.error(f"Error processing image {id}: {e}")
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
    id_list = id_df["ID"].to_list()

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
    optional.add_argument("-ch", "--channl_order", type=str, default="red green blue yellow", help="channel order")
    optional.add_argument("-s", "--frame_size", type=int, default=1024, nargs="+", help="frame size for output image")
    optional.add_argument("-n", "--n_CPU", type=int, default=1, help="number of CPU cores to use")
    optional.add_argument("-m", "--merge_ch0and3", action="store_true")
    args = parser.parse_args()
    setup_logging(args.opdir)
    merge_channels(args)
    return

if __name__ == "__main__":
    main()
