# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.exposure import adjust_log
import argparse
from tqdm import tqdm
from functools import partial
import multiprocessing as mp
import cv2
import logging

def setup_logging(opcsvdir):
    opcsvdir_path = Path(opcsvdir)
    opcsvdir_path.mkdir(exist_ok=True, parents=True)
    log_file_path = opcsvdir_path / 'getstat.log'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def _get_stat(id, ipdir):
    try:
        img = imread(ipdir.joinpath(f"{id}.png"))
        img = img / 255.0
        mean = img.mean(axis=(0, 1))
        std = img.std(axis=(0, 1))
        logging.info(f"Processed image {id} successfully - Mean: {mean}, Std: {std}")
        return (id, mean, std)
    except Exception as e:
        logging.error(f"Error processing image {id}: {e}")
        return (id, None, None)

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

    opcsvpath_all = opcsvdir.joinpath("stats.csv")
    df_results.to_csv(opcsvpath_all, index=False)
    op_mean_avg = df_results["Mean"].mean(axis=0)
    op_std_avg = df_results["Std"].mean(axis=0)
    # print(op_mean_avg, op_std_avg)
    with open(opcsvdir.joinpath("summary.txt"), "w") as f:
        f.write(f"Mean: {op_mean_avg}\n")
        f.write(f"Std: {op_std_avg}\n")
    return opcsvpath_all

# Function to compute average mean and standard deviation from the saved CSV file
def compute_average_stats(csv_file):
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_file)

    # Initialize lists to store channel-wise means and standard deviations
    mean_channels = [[] for _ in range(4)]
    std_channels = [[] for _ in range(4)]

    # Extract mean and std arrays from DataFrame and populate lists
    for index, row in df.iterrows():
        mean_array = np.fromstring(row['Mean'][1:-1], sep=' ')
        std_array = np.fromstring(row['Std'][1:-1], sep=' ')
        
        for i, value in enumerate(mean_array):
            mean_channels[i].append(value)
        
        for i, value in enumerate(std_array):
            std_channels[i].append(value)

    # Compute average mean and standard deviation for each channel
    average_means = [np.mean(channel) for channel in mean_channels]
    average_stds = [np.mean(channel) for channel in std_channels]

    return average_means, average_stds


def main():
    description = "merge three channls into one single RGB file"
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group()
    required.add_argument("-i", "--ipdir", type=str, required=True, help="input dir")
    required.add_argument("-ic", "--ipcsv", type=str, required=True, help="input csv")
    required.add_argument(
        "-oc", "--opcsvdir", type=str, required=True, help="output csv dir"
    )
    optional = parser.add_argument_group()
    optional.add_argument(
        "-n", "--n_CPU", type=int, default=1, help="number of CPU core to use"
    )
    # optional.add_argument("-m", "--merge_ch0and3", action="store_true")
    args = parser.parse_args()
    setup_logging(args.opcsvdir)

    # Compute statistics and save results to CSV
    csv_file = getstat(args)

    # Compute average mean and standard deviation from the saved CSV file
    average_means, average_stds = compute_average_stats(csv_file)

    # Print average mean and standard deviation
    print("\n\n ---------- Average Mean and Standard Deviation for Each Channel ---------- \n")
    for i, (mean, std) in enumerate(zip(average_means, average_stds)):
        print(f"Channel {i+1}: Average Mean = {mean}, Average Std = {std}")

    print("\n")
    return


if __name__ == "__main__":
    main()