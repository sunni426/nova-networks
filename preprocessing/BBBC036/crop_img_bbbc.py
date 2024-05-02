from pathlib import Path
import numpy as np
import pandas as pd
import base64
from pycocotools import _mask as coco_mask
import zlib
import typing as t
import argparse
from skimage.io import imread, imsave
from skimage.measure import label
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


# Function to encode binary mask used to crop the cells into OID challenge encoding ascii text
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


def find_valid_crops(image, masks, save_encode_mask, min_size=(65, 65), max_save=10):
    valid_crops = []
    file_number = 1
    mask_data = []
    unique_masks = np.unique(masks)
    img_height, img_width = image.shape[:2]
    for mask_id in unique_masks:
        if mask_id == 0:
            continue
        mask = masks == mask_id

        where = np.where(mask)
        y_min, y_max = np.min(where[0]), np.max(where[0])
        x_min, x_max = np.min(where[1]), np.max(where[1])

        # Check if the crop size is at least min_sizes
        if (y_max - y_min + 1) < min_size[0] or (x_max - x_min + 1) < min_size[1]:
            continue

        # Exclude cells touching the border
        if y_min == 0 or y_max == img_height - 1 or x_min == 0 or x_max == img_width - 1:
            continue

        # Encode mask and store with file number
        if save_encode_mask:
            encoded_mask = encode_binary_mask(mask)
            mask_data.append((file_number, encoded_mask))

        cropped_cell = np.zeros(
            (y_max - y_min + 1, x_max - x_min + 1, 4), dtype=np.uint8
        )  # RGB

        num_channels = image.shape[-1] 

        for c in range(num_channels):  
            cropped_cell[..., c] = (
                image[y_min : y_max + 1, x_min : x_max + 1, c]
                * mask[y_min : y_max + 1, x_min : x_max + 1]
            )

        valid_crops.append((cropped_cell, file_number))
        file_number += 1

        if max_save != "all" and len(valid_crops) >= max_save:
            break

    if save_encode_mask:
        return valid_crops, mask_data
    else:
        return valid_crops, None


def save_crops(valid_crops, save_dir, id):
    for cropped_image, file_number in valid_crops:
        image_save_path = save_dir.joinpath(f"{id}_cell{str(file_number)}.png")
        imsave(image_save_path, cropped_image, check_contrast=False)
    return


def save_mask_data(mask_data, save_dir, id):
    mask_data_file = save_dir.joinpath(f"{id}_masks.txt")
    with open(mask_data_file, "w") as file:
        for file_number, encoded_mask in mask_data:
            file.write(f"{id}_cell{str(file_number)}: {encoded_mask}\n")
    return

def _crop_image(id, ipimgdir, ipsegdir, opimgdir, opcodedir, max_cellc):
    success = False
    if opcodedir is not None:
        save_encode_mask = True
    else:
        save_encode_mask = False

    try:
        image = imread(ipimgdir.joinpath(f"{id}.png"))
        masks = imread(ipsegdir.joinpath(f"{id}_cp_masks.png"))

        valid_crops, mask_data = find_valid_crops(
            image, masks, save_encode_mask, max_save=max_cellc
        )

        if len(valid_crops) == max_cellc or max_cellc == "all":
            save_crops(valid_crops, opimgdir, id)
            if save_encode_mask:
                save_mask_data(mask_data, opcodedir, id)
            success = True
        else:
            print(f"Warning! Found only {len(valid_crops)} valid cells for file {id}")
    except FileNotFoundError as e:
        print(f"Warning! {e}")
    except Exception as e:
        print(f"Warning! {e}")

    return id, success


def crop_img(args):
    id_df = pd.read_csv(Path(args.ipcsv))
    n_CPU = args.n_CPU
    opimgdir = Path(args.opimgdir)
    opimgdir.mkdir(exist_ok=True, parents=True)
    success_ids = []

    if args.opcodedir is not None:
        opcodedir = Path(args.opcodedir)
        opcodedir.mkdir(exist_ok=True, parents=True)
    else:
        opcodedir = None

    if args.max_cell_count != "all":
        max_cell_count = int(args.max_cell_count)
    else:
        max_cell_count = args.max_cell_count

    crop_img = partial(
        _crop_image,
        ipimgdir=Path(args.ipimgdir),
        ipsegdir=Path(args.ipsegdir),
        opimgdir=opimgdir,
        opcodedir=opcodedir,
        max_cellc=max_cell_count
    )

    # Create a list of ids for each combination of plate, well, and FOV
    id_list = []
    for _, row in id_df.iterrows():
        plate_number = row['Metadata_Plate']
        well_number = row['Metadata_Well']
        for fov in ['s1', 's2', 's3', 's4', 's5', 's6']:
            id_list.append(f"{plate_number}_{well_number}_{fov}")

    if n_CPU != 1:
        with mp.Pool(n_CPU) as pool:
            results = pool.map(crop_img, id_list)
        for id, success in results:
            if success:
                success_ids.append(id)
    else:
        for id in tqdm(id_list):
            success = crop_img(id)
            if success:
                success_ids.append(id)
    
    def is_successful(row):
        plate_well = f"{row['Metadata_Plate']}_{row['Metadata_Well']}"
        return any(f"{plate_well}_{fov}" in success_ids for fov in ['s1', 's2', 's3', 's4', 's5', 's6'])

    filtered_df = id_df[id_df.apply(is_successful, axis=1)]
    new_csv_path = Path(args.ipcsv).with_name(Path(args.ipcsv).stem + '_final.csv')
    filtered_df.to_csv(new_csv_path, index=False)

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
        "-o", "--opimgdir", type=str, required=True, help="output img dir"
    )
    optional = parser.add_argument_group()
    optional.add_argument(
        "--opcodedir", type=str, help="encode binary mask to coco challenge format"
    )
    optional.add_argument(
        "-n", "--n_CPU", type=int, default=1, help="number of CPU core to use"
    )
    optional.add_argument(
        "--max_cell_count",
        type=str,
        default="10",
        help='max cell count per image. default = 10. use "all" to save all cells',
    )
    args = parser.parse_args()
    crop_img(args)
    return


if __name__ == "__main__":
    main()