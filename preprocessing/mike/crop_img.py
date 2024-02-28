from pathlib import Path
import numpy as np
import pandas as pd
import base64
from pycocotools import _mask as coco_mask
import zlib
import typing as t
import argparse
from skimage.io import imread, imsave
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


# Function to encode binary mask (red channel) used to crop the cells into OID challenge encoding ascii text
def encode_binary_mask(mask: np.ndarray) -> t.Text:
    if mask.dtype != np.bool_:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s"
            % mask.dtype
        )

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" % mask.shape
        )

    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1).astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode("utf-8")


def find_valid_crops(image, masks, min_size=(100, 100), max_save=10):
    valid_crops = []
    file_number = 1
    unique_masks = np.unique(masks)
    for mask_id in unique_masks:
        if mask_id == 0:
            continue
        mask = masks == mask_id

        where = np.where(mask)
        y_min, y_max = np.min(where[0]), np.max(where[0])
        x_min, x_max = np.min(where[1]), np.max(where[1])

        # Check if the crop size is at least 100x100
        if (y_max - y_min + 1) < min_size[0] or (x_max - x_min + 1) < min_size[1]:
            continue

        cropped_cell = np.zeros(
            (y_max - y_min + 1, x_max - x_min + 1, 3), dtype=np.uint8
        )  # RGB
        for c in range(3):  # Copy color channels
            cropped_cell[..., c] = (
                image[y_min : y_max + 1, x_min : x_max + 1, c]
                * mask[y_min : y_max + 1, x_min : x_max + 1]
            )

        valid_crops.append((cropped_cell, file_number))
        file_number += 1

        if len(valid_crops) >= max_save:
            break

    return valid_crops


def save_crops(valid_crops, save_dir, orig_filename):
    file_c_digit = len(str(len(valid_crops)))
    for cropped_image, file_number in valid_crops:
        image_save_path = save_dir.parent.joinpath(
            f"{orig_filename}_cell{str(file_number).zfill(file_c_digit)}.png"
        )
        imsave(image_save_path, cropped_image)
    return


def _crop_image(id, ipimgdir, ipsegdir, opdir, max_cellc):
    image = imread(ipimgdir.joinpath(f"{id}.png"))
    masks = imread(ipsegdir.joinpath(f"{id}_cp_masks.png"))
    opimgpath = opdir.joinpath(f"{id}.png")
    valid_crops = find_valid_crops(image, masks, max_save=max_cellc)

    if len(valid_crops) == max_cellc:
        save_crops(valid_crops, opimgpath, id)
    else:
        print(
            f"Warning: Expected to save {max_cellc} cells, but found only {len(valid_crops)} valid cells for file {opimgpath.stem}"
        )
    return


def crop_img(args):
    id_df = pd.read_csv(Path(args.ipcsv))
    id_list = id_df["ID"].to_list()
    n_CPU = args.n_CPU
    opdir = Path(args.opdir)
    opdir.mkdir(exist_ok=True, parents=True)

    _crop_img = partial(
        _crop_image,
        ipimgdir=Path(args.ipimgdir),
        ipsegdir=Path(args.ipsegdir),
        opdir=opdir,
        max_cellc=args.max_cell_count,
    )

    if n_CPU != 1:
        pbar = tqdm(total=len(id_list))
        results = []

        def pbar_update(results):
            pbar.update(1)

        pool = mp.Pool(n_CPU)
        for idx, id in enumerate(id_list):
            pool.apply_async(_crop_img, (id,), callback=pbar_update)
        pool.close()
        pool.join()
    else:
        for idx, id in tqdm(enumerate(id_list)):
            _crop_img(id)

    return


def main():
    description = "crop single cell images"
    parser = argparse.ArgumentParser(description=description)
    required = parser.add_argument_group()
    required.add_argument("-ic", "--ipcsv", type=str, required=True, help="input csv")
    required.add_argument(
        "-ii", "--ipimgdir", type=str, required=True, help="input image dir"
    )
    required.add_argument(
        "-is", "--ipsegdir", type=str, required=True, help="input seg dir"
    )
    required.add_argument(
        "-o", "--opdir", type=str, required=True, help="output img dir"
    )
    # required.add_argument('-c', type=str, required=True, help='output csv dir')
    optional = parser.add_argument_group()
    optional.add_argument(
        "-n", "--n_CPU", type=int, default=1, help="number of CPU core to use"
    )
    optional.add_argument(
        "--max_cell_count", type=int, default=10, help="max cell count per image"
    )
    args = parser.parse_args()
    crop_img(args)
    return


if __name__ == "__main__":
    main()