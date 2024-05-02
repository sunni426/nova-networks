# %%
from pathlib import Path
import argparse
from skimage.io import imread, imsave
from cellpose import models, plot, utils
from cellpose import io
import multiprocessing as mp
from tqdm import tqdm
import torch

def load_checkpoints(checkpoint_path):
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as file:
            processed_files = file.read().splitlines()
        return set(processed_files)
    else:
        return set()

def save_checkpoint(checkpoint_path, filepath):
    with open(checkpoint_path, 'a') as file:
        file.write(f"{filepath}\n")

# %%
def cellpose_seg(args):
    input_path = Path(args.i)
    filepaths = list(input_path.glob("*.png")) 
    opdir = Path(args.o)
    opdir.mkdir(parents=True, exist_ok=True)
    n = args.n_trunk

    checkpoint_path = opdir / 'processed_files.txt'
    processed_files = load_checkpoints(checkpoint_path)
    filepaths = [fp for fp in filepaths if str(fp) not in processed_files]

    filepaths_split = [filepaths[i : i + n] for i in range(0, len(filepaths), n)]

    model = models.Cellpose(
        gpu=True, model_type="cyto2", device=torch.device(f"cuda:{args.gpu_device}")
    )
    for batch_filepaths in tqdm(filepaths_split):
        imgs = [imread(filepath) for filepath in batch_filepaths]
        masks, flows, styles, diams = model.eval(
            imgs,
            diameter=None,
            channels=[1, 3],
            flow_threshold=0.8,
            cellprob_threshold=0.0,
            batch_size=args.batch_size,
        )
        opdatapath = [opdir.joinpath(f"{filepath.stem}.npy") for filepath in batch_filepaths]
        opsegpath = [opdir.joinpath(f"{filepath.stem}.png") for filepath in batch_filepaths]
        io.masks_flows_to_seg(imgs, masks, flows, diams, opdatapath)
        io.save_to_png(imgs, masks, flows, opsegpath)

        for filepath in batch_filepaths:
            save_checkpoint(checkpoint_path, filepath)


def main():
    description = "Cellpose Segmentation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", type=str, required=True, help="input dir")
    parser.add_argument("-o", type=str, required=True, help="output dir")
    parser.add_argument("-g", "--gpu_device", type=int, default="0", help="gpu device")
    parser.add_argument("-n", "--n_trunk", type=int, default="10", help="number of trunk")
    parser.add_argument("-b", "--batch_size", type=int, default="256", help="batch size")
    args = parser.parse_args()
    cellpose_seg(args)
    return


if __name__ == "__main__":
    main()
# %%
