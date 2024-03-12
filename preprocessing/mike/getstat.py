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
import cv2


def _get_stat(id, ipdir):
    img = imread(ipdir.joinpath(f"{id}.png"))
    mean = img.mean(axis=(0, 1))
    std = img.std(axis=(0, 1))
    return (id, mean, std)


def getstat(args):
    ipdir = Path(args.ipdir)
    ipcsvpath = Path(args.ipcsv)
    opcsvdir = Path(args.opcsvdir)
    opcsvdir.mkdir(exist_ok=True, parents=True)
    n_CPU = args.n_CPU

    # load id list
    id_df = pd.read_csv(ipcsvpath)
    id_list = id_df["ID"].to_list()
    # id_list = id_list[:10]
    _getstat = partial(
        _get_stat,
        ipdir=ipdir,
    )

    results = []
    if n_CPU != 1:
        pbar = tqdm(total=len(id_list))
        # results = []

        def pbar_update(result):
            pbar.update(1)
            results.append(result)

        pool = mp.Pool(n_CPU)
        for idx, id in enumerate(id_list):
            pool.apply_async(_getstat, (id,), callback=pbar_update)

        pool.close()
        pool.join()
    else:
        for idx, id in tqdm(enumerate(id_list)):
            results.append(_getstat(id))
    print(results)
    df_results = pd.DataFrame(results, columns=["ID", "Mean", "Std"])

    opcsvpath_all = opcsvdir.joinpath("all.csv")
    df_results.to_csv(opcsvpath_all, index=False)
    op_mean_avg = df_results["Mean"].mean(axis=0)
    op_std_avg = df_results["Std"].mean(axis=0)
    # print(op_mean_avg, op_std_avg)
    with open(opcsvdir.joinpath("summary.txt"), "w") as f:
        f.write(f"Mean: {op_mean_avg}\n")
        f.write(f"Std: {op_std_avg}\n")
    return


def main():
    description = "merge three channls into one single RGB file"
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group()
    required.add_argument("-i", "--ipdir", type=str, required=True, help="input dir")
    required.add_argument("-ic", "--ipcsv", type=str, required=True, help="input csv")
    required.add_argument(
        "-oc", "--opcsvdir", type=str, required=True, help="output img dir"
    )
    optional = parser.add_argument_group()
    optional.add_argument(
        "-n", "--n_CPU", type=int, default=1, help="number of CPU core to use"
    )
    # optional.add_argument("-m", "--merge_ch0and3", action="store_true")
    args = parser.parse_args()
    getstat(args)

    return


if __name__ == "__main__":
    main()
