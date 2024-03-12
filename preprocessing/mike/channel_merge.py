# %%
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


def _merge_ch(id, ipdir, opdir, chcolors, opframesize, merge03):
    img = []
    for chcolor in chcolors:
        img_tmp = imread(ipdir.joinpath(f"{id}_{chcolor}.png"))
        img.append(img_tmp)
    img = np.stack(img, axis=-1)

    if merge03:
        # merge yellow and red
        if len(chcolor) > 3:
            img[0] = img[0] + img[3]
            img[0] = np.clip(img[0], 0, 255)
            img = img[:, :, :3]

    img = adjust_log(img, 1)
    img = resize(
        img, (opframesize[0], opframesize[1], len(chcolors)), anti_aliasing=True
    )
    img = img * 255
    img = img.astype(np.uint8)
    opimgpath = opdir.joinpath(f"{id}.png")
    imsave(opimgpath, img, check_contrast=False)

    return


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

    # load id list
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
        results = []

        def pbar_update(results):
            pbar.update(1)

        pool = mp.Pool(n_CPU)
        for idx, id in enumerate(id_list):
            pool.apply_async(_merge_channels, (id,), callback=pbar_update)

        pool.close()
        pool.join()
    else:
        for idx, id in tqdm(enumerate(id_list)):
            _merge_channels(id)

    return


def main():
    description = "merge three channls into one single RGB file"
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group()
    required.add_argument("-i", "--ipdir", type=str, required=True, help="input dir")
    required.add_argument("-ic", "--ipcsv", type=str, required=True, help="input csv")
    required.add_argument(
        "-o", "--opdir", type=str, required=True, help="output img dir"
    )
    # required.add_argument('-c', type=str, required=True, help='output csv dir')
    optional = parser.add_argument_group()
    optional.add_argument(
        "-ch",
        "--channl_order",
        type=str,
        default="red green blue yellow",
        help="channel order",
    )
    optional.add_argument(
        "-s",
        "--frame_size",
        type=int,
        default=1024,
        nargs="+",
        help="frame size for output image",
    )
    optional.add_argument(
        "-n", "--n_CPU", type=int, default=1, help="number of CPU core to use"
    )
    optional.add_argument("-m", "--merge_ch0and3", action="store_true")
    args = parser.parse_args()
    merge_channels(args)
    return


if __name__ == "__main__":
    main()
