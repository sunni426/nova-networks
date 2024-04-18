# %%
import re
from pathlib import Path
import argparse
from skimage.io import imread, imsave
from cellpose import models, plot, utils
from cellpose import io
import multiprocessing as mp
from tqdm import tqdm
import torch


# %%
def cellpose_seg(args):
    filepaths = list(Path(args.i).glob("*.png"))
    # filepaths = [Path(x) for x in filepaths]
    opdir = Path(args.o)
    opdir.mkdir(parents=True, exist_ok=True)

    oppaths = [
        opdir.joinpath(f"{x.stem}_seg").with_suffix(".npy") for x in tqdm(filepaths)
    ]
    print(len(oppaths))
    oppaths = [x for x in tqdm(oppaths) if not x.is_file()]
    filepaths_proc = [
        Path(args.i).joinpath(re.sub("_seg", "", x.stem)).with_suffix(".png")
        for x in tqdm(oppaths)
    ]
    print(len(filepaths_proc))

    n = args.n_trunk
    filepaths_split = [filepaths[i : i + n] for i in range(0, len(filepaths_proc), n)]

    model = models.Cellpose(
        gpu=True, model_type="cyto2", device=torch.device(f"cuda:{args.gpu_device}")
    )
    for filepaths in tqdm(filepaths_split):
        imgs = [imread(filepath) for filepath in filepaths]
        masks, flows, styles, diams = model.eval(
            imgs,
            diameter=150,
            channels=[1, 3],
            flow_threshold=0.8,
            cellprob_threshold=0.0,
            batch_size=args.batch_size,
        )
        opdatapath = [opdir.joinpath(f"{filepath.stem}.npy") for filepath in filepaths]
        opsegpath = [opdir.joinpath(f"{filepath.stem}.png") for filepath in filepaths]
        io.masks_flows_to_seg(imgs, masks, flows, diams, opdatapath)
        io.save_to_png(imgs, masks, flows, opsegpath)

    return


def main():
    description = "Cellpose segmentation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", type=str, required=True, help="input dir")
    parser.add_argument("-o", type=str, required=True, help="output dir")
    parser.add_argument("-g", "--gpu_device", type=int, default="0", help="gpu device")
    parser.add_argument(
        "-n", "--n_trunk", type=int, default="10", help="number of trunk"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default="256", help="batch size"
    )
    args = parser.parse_args()
    cellpose_seg(args)
    return


if __name__ == "__main__":
    main()
