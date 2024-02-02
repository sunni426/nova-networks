cd predict/data_process
python s1_generate_cellmask.py --gpus 0 --dataset test
python s2_generate_cellimage.py --n_cpu 16 --dataset train --cell_type 1 --image_size 512
python s3_resize_images.py --image_size 128 --suffix png --dataset train_cell_v1

python generate_gradcam.py --module cls_inception_v3_gradcam --model_name cls_inception_v3_cbam --fold 0 --batch_size 4 --in_channels 4 --num_classes 19 --image_size 128 --model_dir d0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds --model_epoch 12.00_ema --dataset test --cell_type 1 --overwrite 1 --gpus 0

cd <data_path> #rename files if needed (rename *_mask.png to *_cellmask.png)
rename -v 's/mask/cellmask/' *.png
