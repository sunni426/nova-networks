1.Environment and file descriptions
1.1 The basic Runtime Environment is python3.6, pytorch1.6.0, you can refer to requriements.txt to set up your environment.

2. Prepare data
2.1 Please config your local directory in config.config.py(for export cell features) and notebook(for UMAP)
2.2 Please unpack train.zip to DATA_DIR/images (provided by Kaggle, is not included in my file).
2.3 Please move train.csv to DATA_DIR/raw (provided by Kaggle, is not included in my file)
2.4 Please move segmentation models to PRETRAINED_DIR (provided by Kaggle forum, is not included in my file)
https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg

3. Preprocess data
3.1 Generate cell mask data, DATASET is train or other datasets
process :
    cd CODE_DIR/data_process/
    python s1_generate_cellmask.py --gpus 0 --dataset DATASET
input :
    PRETRAINED_DIR/cell-model.pth
    PRETRAINED_DIR/nuclei-model.pth
    DATA_DIR/raw/train.csv
    DATA_DIR/images/{DATASET}
output :
    DATA_DIR/inputs/{DATASET}.csv
    DATA_DIR/mask/{DATASET}
    DATA_DIR/mask/{DATASET}.csv
### python s2_generate_cellimage.py --n_cpu 16 --dataset test --cell_type 1 --image_size 512
### python s3_resize_images.py --image_size 128 --suffix png --dataset test_cell_v1
3.2 Generate cropped cell images and resize to 128x128
process :
    cd CODE_DIR/data_process/
    python s2_generate_cellimage.py --n_cpu 16 --dataset DATASET --cell_type 1 --image_size 128
input :
    DATA_DIR/inputs/{DATASET}.csv
    DATA_DIR/images/{DATASET}
    DATA_DIR/mask/{DATASET}
output :
    DATA_DIR/images/{DATASET}_cell_v1_png_i128x128

4. Predict InceptionV3 cell model (private 0.547 public0.565)
process :
    cd CODE_DIR
    python predict_cell.py  --module cls_inception_v3 --model_name cls_inception_v3_cbam --fold 0 --batch_size 4 --in_channels 4 --num_classes 19 --image_size 128 --model_dir d0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds --model_epoch 12.00_ema --dataset train --cell_type 1 --overwrite 1 --gpus 0,1
input :
    DATA_DIR/mask/{DATASET}.csv
    DATA_DIR/images/{DATASET}_cell_v1_png_i128x128
    RESULT_DIR/models/d0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds/fold0/12.00_ema.pth
output :
    RESULT_DIR/submissions/d0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds/fold0/epoch_12.00_ema/cell_result_{DATASET}_default_cell_v1.npy
