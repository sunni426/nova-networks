from cProfile import label
import pandas as pd
from object_detection.metrics import oid_challenge_evaluation_utils as utils
import tqdm
import imageio
import os
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib


def encode_binary_mask(mask: np.ndarray) -> t.Text:
  """Converts a binary mask into OID challenge encoding ascii text."""

  # check input mask --
  if mask.dtype != np.bool:
    raise ValueError(
        "encode_binary_mask expects a binary mask, received dtype == %s" %
        mask.dtype)

  mask = np.squeeze(mask)
  if len(mask.shape) != 2:
    raise ValueError(
        "encode_binary_mask expects a 2d mask, received shape == %s" %
        mask.shape)

  # convert input mask to expected COCO API input --
  mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
  mask_to_encode = mask_to_encode.astype(np.uint8)
  mask_to_encode = np.asfortranarray(mask_to_encode)

  # RLE encode mask --
  encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

  # compress and base64 encoding --
  binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
  base64_str = base64.b64encode(binary_str)
  return base64_str

if False:
    if True:  # Formatting the solution files, only need to be done once!
        all_annotations_hpa = pd.read_csv(
            "/home/trangle/HPA_SingleCellClassification/GT/_solution.csv_"
        )
        all_annotations = pd.DataFrame()
        for i, row in all_annotations_hpa.iterrows():
            pred_string = row.PredictionString.split(" ")
            for k in range(0, len(pred_string), 7):
                boxes = utils._get_bbox(
                    pred_string[k + 6], row.ImageWidth, row.ImageWidth
                )  # ymin, xmin, ymax, xmax
                line = {
                    "ImageID": row.ID,
                    "ImageWidth": row.ImageWidth,
                    "ImageHeight": row.ImageHeight,
                    "ConfidenceImageLabel": 1,
                    "LabelName": pred_string[k],
                    "XMin": boxes[1],
                    "YMin": boxes[0],
                    "XMax": boxes[3],
                    "YMax": boxes[2],
                    "IsGroupOf": pred_string[k + 5],
                    "Mask": pred_string[k + 6],
                }
                """
              binary_mask = decodeToBinaryMask(pred_string[k+6], row.ImageWidth, row.ImageHeight)
              plt.figure()
              plt.imshow(binary_mask)
              """
                all_annotations = all_annotations.append(line, ignore_index=True)
            print(f"{i}/{len(all_annotations_hpa)}, {len(all_annotations)} cells")
        all_annotations_hpa.to_csv(
            "/home/trangle/HPA_SingleCellClassification/all_annotations.csv",
            index=False,
        )


f = open("/home/trangle/HPA_SingleCellClassification/GT/all_annotations_raw.csv", "a+")
f.write("ImageID,ImageWidth,ImageHeight,ConfidenceImageLabel,LabelName,XMin,YMin,XMax,YMax,IsGroupOf,Mask\n")
mask_dir = "/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/data"
labels = pd.read_csv("/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/labels.csv")
labels["ImageID"] = [id.split("_")[0] for id in labels.ID]
labels["cell_idx"] = [id.split("_")[1] for id in labels.ID]

imlist = list(set(labels.ImageID))
for _, img_id in tqdm.tqdm(enumerate(imlist), total=len(imlist)):
    image_df = labels[labels.ImageID == img_id]
    mask = imageio.imread(os.path.join(mask_dir, img_id+"_mask.png"))
    width = mask.shape[0]
    height = mask.shape[1]
    for i, row in image_df.iterrows():
        cell_idx = row.cell_idx
        m = mask==int(cell_idx)
        if m.max() == False:
            print(f'idx in mask {int(cell_idx)+1}',row)
        rle = encode_binary_mask(m)
        boxes = utils._get_bbox(rle, width, height)
        labs = row.Label.split("|")
        for l in labs:
            line = f"{img_id},{width},{height},1,{l},{boxes[1]},{boxes[0]},{boxes[3]},{boxes[2]},0,{rle}\n"
            f.write(line)
            #print(line)
f.close()
