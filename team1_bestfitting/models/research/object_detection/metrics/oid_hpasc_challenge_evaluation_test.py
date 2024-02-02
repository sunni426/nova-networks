# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs evaluation using OpenImages groundtruth and predictions.

Uses Open Images Challenge 2018, 2019 metrics

Example usage:
python models/research/object_detection/metrics/oid_od_challenge_evaluation.py \
    --input_annotations_boxes=/path/to/input/annotations-human-bbox.csv \
    --input_annotations_labels=/path/to/input/annotations-label.csv \
    --input_class_labelmap=/path/to/input/class_labelmap.pbtxt \
    --input_predictions=/path/to/input/predictions.csv \
    --output_metrics=/path/to/output/metric.csv \
    --input_annotations_segm=[/path/to/input/annotations-human-mask.csv] \

If optional flag has_masks is True, Mask column is also expected in CSV.

CSVs with bounding box annotations, instance segmentations and image label
can be downloaded from the Open Images Challenge website:
https://storage.googleapis.com/openimages/web/challenge.html
The format of the input csv and the metrics itself are described on the
challenge website as well.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from absl import app
from absl import flags
import pandas as pd
from google.protobuf import text_format

from object_detection.metrics import io_utils
from object_detection.metrics import oid_challenge_evaluation_utils as utils
from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import object_detection_evaluation
import time
import tqdm
import os
import cProfile, pstats, io
from pstats import SortKey
from multiprocessing import Process, Lock

flags.DEFINE_string('all_annotations', None,
                    'File with groundtruth boxes and label annotations.')
flags.DEFINE_string(
    'input_predictions', None,
    """File with detection predictions; NOTE: no postprocessing is applied in the evaluation script."""
)
flags.DEFINE_string('input_class_labelmap', None,
                    'Open Images Challenge labelmap.')
flags.DEFINE_string('output_metrics', None, 'Output file with csv metrics.')
flags.DEFINE_string(
    'input_annotations_segm', None,
    'File with groundtruth instance segmentation annotations [OPTIONAL].')

FLAGS = flags.FLAGS


def _load_labelmap(labelmap_path):
  """Loads labelmap from the labelmap path.

  Args:
    labelmap_path: Path to the labelmap.

  Returns:
    A dictionary mapping class name to class numerical id
    A list with dictionaries, one dictionary per category.
  """

  label_map = string_int_label_map_pb2.StringIntLabelMap()
  with open(labelmap_path, 'r') as fid:
    label_map_string = fid.read()
    text_format.Merge(label_map_string, label_map)
  labelmap_dict = {}
  categories = []
  for item in label_map.item:
    labelmap_dict[item.name] = item.id
    categories.append({'id': item.id, 'name': item.name})
  return labelmap_dict, categories


def main(unused_argv):
  s = time.time()
  flags.mark_flag_as_required('all_annotations')
  flags.mark_flag_as_required('input_predictions')
  flags.mark_flag_as_required('input_class_labelmap')
  flags.mark_flag_as_required('output_metrics')

  is_instance_segmentation_eval = False
  resume = False

  if False: #Formatting the solution files, only need to be done once!!
      all_annotations_hpa = pd.read_csv("/home/trangle/HPA_SingleCellClassification/GT/_solution.csv_")
      f = open(FLAGS.all_annotations, "a+")
      f.write("ImageID,ImageWidth,ImageHeight,ConfidenceImageLabel,LabelName,XMin,YMin,XMax,YMax,IsGroupOf,Mask\n")
      # Testing with 10 images
      #imlist = list(set(all_annotations_hpa.ID))[:10]
      #all_annotations_hpa = all_annotations_hpa[all_annotations_hpa.ID.isin(imlist)]
      for i, row in all_annotations_hpa.iterrows():
          pred_string = row.PredictionString.split(" ")
          for k in range(0, len(pred_string), 7):
              boxes = utils._get_bbox(pred_string[k+6], row.ImageWidth,row.ImageWidth) # ymin, xmin, ymax, xmax
              if pred_string[k+6] == '-1':
                continue
              line = f"{row.ID},{row.ImageWidth},{row.ImageHeight},1,{pred_string[k]},{boxes[1]},{boxes[0]},{boxes[3]},{boxes[2]},{pred_string[k+5]},{pred_string[k+6]}\n"
              print(line)
              f.write(line)
      f.close()
      
  all_annotations = pd.read_csv(FLAGS.all_annotations, header=0)
  all_annotations['ImageHeight'] = all_annotations['ImageHeight'].astype(str).astype(int)
  all_annotations['ImageWidth'] = all_annotations['ImageWidth'].astype(int)
  private_df = pd.read_csv("/home/trangle/HPA_SingleCellClassification/GT/labels_privatetest.csv")
  imlist = list(set(private_df.ImageID))
  # Testing with last 3 images
  # imlist = list(set(all_annotations.ImageID))[-3:]
  all_annotations = all_annotations[all_annotations.ImageID.isin(imlist)]

  class_label_map, categories = _load_labelmap(FLAGS.input_class_labelmap)
  #class_label_map, categories = _load_labelmap("/home/trangle/HPA_SingleCellClassification/models/research/object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt")
  challenge_evaluator = (
      object_detection_evaluation.OpenImagesChallengeEvaluator(
          categories, evaluate_masks=is_instance_segmentation_eval, matching_iou_threshold=0.6))
  if resume:
    if os.path.exists(os.path.join(os.path.dirname(FLAGS.output_metrics), 'obj', 'internal_state.pkl')):
      current_state = io_utils.load_obj(os.path.dirname(FLAGS.output_metrics), 'internal_state')
      challenge_evaluator._evaluation.merge_internal_state(current_state)
      images_processed = io_utils.load_obj(os.path.dirname(FLAGS.output_metrics), 'images_processed')
      print(current_state)
    else:
      print('no internal state file, start from scratch')
      images_processed = []
  else:
    images_processed = []

  if not os.path.exists(FLAGS.input_predictions.replace(".csv","_formatted.csv")):
    submissions = pd.read_csv(FLAGS.input_predictions)      
    f = open(FLAGS.input_predictions.replace(".csv","_formatted.csv"), "a+")
    f.write("ImageID,ImageWidth,ImageHeight,LabelName,Score,Mask\n")
    for i, row in tqdm.tqdm(submissions.iterrows(),total=submissions.shape[0]):
        try:
            pred_string = row.PredictionString.split(" ")
            for k in range(0, len(pred_string), 3):
                  label = pred_string[k]
                  conf = pred_string[k + 1]
                  rle = pred_string[k + 2]
                  line = f"{row.ID},{row.ImageWidth},{row.ImageHeight},{str(label)},{conf},{rle}\n"
                  f.write(line)
        except:
            continue

  #pr = cProfile.Profile()
  #pr.enable()
  all_predictions= pd.read_csv(FLAGS.input_predictions.replace(".csv","_formatted.csv"))
  all_predictions['LabelName'] = [str(l) for l in all_predictions.LabelName]

  for _, groundtruth in tqdm.tqdm(enumerate(all_annotations.groupby('ImageID')), total=all_annotations.ImageID.nunique()):
    #if images_processed == 20:
    #  pass
    image_id, image_groundtruth = groundtruth
    if image_id in images_processed:
      continue
    groundtruth_dictionary = utils.build_groundtruth_dictionary(
        image_groundtruth, class_label_map)
    challenge_evaluator.add_single_ground_truth_image_info(
        image_id, groundtruth_dictionary)
    #print('evaluable_labels', challenge_evaluator._evaluatable_labels[image_id])

    prediction_dictionary = utils.build_predictions_dictionary(
        all_predictions.loc[all_predictions['ImageID'] == image_id],
        class_label_map)
    
    challenge_evaluator.add_single_detected_image_info(image_id,
                                                       prediction_dictionary)
    #print('label_id_offset', challenge_evaluator._label_id_offset)
    #print(challenge_evaluator._evaluation.get_internal_state().num_gt_instances_per_class)
    #print(challenge_evaluator.scores_per_class)
    images_processed += [image_id]
    io_utils.save_obj(images_processed, os.path.dirname(FLAGS.output_metrics), 'images_processed')
    io_utils.save_obj(challenge_evaluator._evaluation.get_internal_state(), os.path.dirname(FLAGS.output_metrics), 'internal_state')
  metrics = challenge_evaluator.evaluate()
  #print('corlocs', challenge_evaluator._evaluate_corlocs)
  with open(FLAGS.output_metrics, 'w') as fid:
    io_utils.write_csv(fid, metrics)

  #pr.disable()
  #s = io.StringIO()
  #sortby = SortKey.CUMULATIVE
  #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
  #ps.print_stats()
  #print(s.getvalue())

  print(f'Finished in {(time.time() - s)/3600} hour')

if __name__ == '__main__':
  app.run(main)