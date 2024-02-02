from operator import index
import sys

sys.path.insert(0, '..')
import argparse
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from importlib import import_module
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import glob
import torch
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from utilities.augment_util import *
from utilities.model_util import load_pretrained
from dataset.cell_cls_dataset import CellClsDataset as HPA2021Dataset
from dataset.cell_cls_dataset import cls_collate_fn as collate_fn
cudnn.benchmark = True

import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def initialize_environment(args):
    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # COMMON
    args.device = 'cuda' if args.gpus else 'cpu'
    args.can_print = True
    args.seed = seed
    args.debug = False

    # MODEL
    args.scheduler = None
    args.loss = None
    args.pretrained = False

    # DATASET
    args.image_size = args.image_size.split(',')
    if len(args.image_size) == 1:
        args.image_size = [int(args.image_size[0]), int(args.image_size[0])]
    elif len(args.image_size) == 2:
        args.image_size = [int(args.image_size[0]), int(args.image_size[1])]
    else:
        raise ValueError(','.join(args.image_size))

    args.num_workers = 4
    args.split_type = 'random'
    args.suffix = 'png'
    args.augments = args.augments.split(',')

    if args.can_print:
        if args.gpus:
            print(f'use gpus: {args.gpus}')
        else:
            print(f'use cpu')
    if args.gpus:
      os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.model_fpath = f'{DIR_CFGS.MODEL_DIR}/{args.model_dir}/fold{args.fold}/{args.model_epoch}.pth'
    args.output_dir = f'{DIR_CFGS.DATA_DIR}/1st_cams/test_df_cams'
    args.feature_dir = f'{DIR_CFGS.FEATURE_DIR}/{args.model_dir}/fold{args.fold}/epoch_{args.model_epoch}'

    if args.can_print:
        print(f'output dir: {args.output_dir}')
        print(f'feature dir: {args.feature_dir}')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.feature_dir, exist_ok=True)
    return args

def load_model(args):
    image_size = args.image_size
    args.image_size = image_size[0]
    model = import_module(f'net.{args.module}').get_model(args)[0]
    args.image_size = image_size
    model = load_pretrained(model, args.model_fpath, strict=True, can_print=args.can_print)
    model = model.eval().to(args.device)
    return model

def generate_dataloader(args):
    test_dataset = HPA2021Dataset(
      args,
      transform=None,
      dataset=args.dataset,
      is_training=False,
    )
    test_dataset.set_part(part_start=args.part_start, part_end=args.part_end)
    _ = test_dataset[0]
    test_loader = DataLoader(
      test_dataset,
      sampler=SequentialSampler(test_dataset),
      batch_size=1,
      drop_last=False,
      num_workers=args.num_workers,
      pin_memory=False,
      collate_fn=collate_fn,
    )
    print(f'num: {test_dataset.__len__()}')
    return test_loader

def iou(a,b):
    intersection = np.intersect1d(a, b)
    union = np.union1d(a, b)
    iou = intersection.shape[0] / union.shape[0]
    return iou

def generate_cam(args, test_loader, model):
    model = load_model(args) if model is None else model
    #print(model.fc_layers[2],model.fc_layers[4])
    #print(model.backbone.layer4[-1])
    test_loader = generate_dataloader(args) if test_loader is None else test_loader
    target_layers = [model.backbone.layer4[-1]]#[model.att_module.conv_after_concat] #[model.backbone.layer4[-1]] #[model.fc_layers[-2]] 
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=('cuda' == args.device))
    
    #resutls = []
    augment='default' # default == nothing
    F = open(f"{args.output_dir}/result_df2.csv", "a+")
    print(f'writing to {args.output_dir}/result_df2.csv')
    F.write("image_id,cell_id,label,prob,ssim,mse,binarized_ssim,binarized_mse\n")
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc=f'cell {augment}'):
        
        outputs = model(Variable(iter_data['image'].to(args.device)))
        logits = outputs
        probs = torch.sigmoid(logits)
        probs = probs.to('cpu').detach().numpy()[0]
        #top3 = np.argsort(probs)
        inp = iter_data['image'].to(args.device)
        rgb_img = np.array(iter_data['image'][0][0:3,:,:]*255).astype(np.uint8).transpose(1, 2, 0)
        # channels:['red', 'green', 'blue', 'yellow'] or MT, protein, nu, ER
        # cv2 is in format BGR
        bgr_img = rgb_img[...,[2,1,0]]
        _,bgreen = cv2.threshold(bgr_img[:,:,1], 20, 255, cv2.THRESH_BINARY)
        #cv2.imwrite(f"{args.output_dir}/{iter_data['ID'][0]}.png", bgr_img)
        for v,k in LABEL_TO_ALIAS.items():
            prob = np.round(probs[v],2)
            grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(v)], aug_smooth=True)

            """
            if v in top3:
                print(prob,grayscale_cam.mean(),grayscale_cam.max())
            #if prob > 0.3:
                visualization = show_cam_on_image(bgr_img/255, grayscale_cam[0], use_rgb=False)
                sidebyside = np.r_[bgr_img,visualization]
                cv2.imwrite(f"{args.output_dir}/{iter_data['ID'][0]}_{iter_data['index'][0]}_{k}_{prob:.2f}.png", sidebyside)
            """
            m = mse(bgr_img[:,:,1], grayscale_cam.squeeze())
            s = ssim(bgr_img[:,:,1], grayscale_cam.squeeze(), mode='valid')
            
            # converting to its binary form cv2.threshold(img, thres1, thres2, thres_type)
            _,bcam = cv2.threshold(grayscale_cam.squeeze(), 0.1, 255, cv2.THRESH_BINARY)
            bs = ssim(bgreen, bcam, mode='valid')
            bm = mse(bgreen, bcam)
            
            #print(bs,s)
            if bs > 0.99:
                sidebyside = np.r_[bgreen,bcam]
                cv2.imwrite(f"{args.output_dir}/{iter_data['ID'][0]}_{iter_data['index'][0]}_{k}_{prob:.2f}.png", sidebyside)
            line = f"{iter_data['ID'][0]},{iter_data['maskid'][0]},{k},{probs[v]},{s},{m},{bs},{bm}\n"
            F.write(line)
    F.close()
            #resutls.append((iter_data['ID'][0], iter_data['index'][0], k, probs[v], s, m))
    #resutls = pd.DataFrame(resutls, columns=["image_id", "cell_id", "Label", "prob", "ssim", "mse"])
    #resutls.to_csv(f"{args.output_dir}/result_df.csv", index=False)

def generate_cams_sclabel(args, test_loader, model):
    labels_test = pd.read_csv('/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/labels.csv')
    labels_test = labels_test[labels_test.Label.isin(list([str(f) for f in range(19)]))]
    '''
    # already done
    df = pd.read_csv(f"{args.output_dir}/result_df_singlelabel.csv")
    df['ID'] = ['_'.join((row.image_id, str(row.cell_id))) for i,row in df.iterrows()]
    labels_test = labels_test[~labels_test.ID.isin(df.ID)]
    '''
    model = load_model(args) if model is None else model

    test_loader = generate_dataloader(args) if test_loader is None else test_loader
    target_layers = [model.backbone.layer4[-1]]#[model.att_module.conv_after_concat] #[model.backbone.layer4[-1]] #[model.fc_layers[-2]] 
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=('cuda' == args.device))
    
    augment='default'
    F = open(f"{args.output_dir}/result_df_all.csv", "a+")
    print(f'Writing to {args.output_dir}/result_df_all.csv')
    F.write("image_id,cell_id,label,prob,ssim,mse,binarized_ssim,binarized_mse,pearsonr_masked,pval_masked,pearsonr,pval,iou_all,iou_masked\n")
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc=f'cell {augment}'):

        #if "_".join((str(iter_data['ID'][0]),str(iter_data['maskid'][0]))) not in labels_test.ID.values:
        #    continue
        outputs = model(Variable(iter_data['image'].to(args.device)))
        logits = outputs
        probs = torch.sigmoid(logits)
        probs = probs.to('cpu').detach().numpy()[0]
        inp = iter_data['image'].to(args.device)
        compare_region = np.where(np.sum(iter_data['image'][0].numpy(), axis=0).flatten()>0)[0]
        rgb_img = np.array(iter_data['image'][0][0:3,:,:]*255).astype(np.uint8).transpose(1, 2, 0)
        #grayscale_cams = cam(input_tensor=inp, targets=[ClassifierOutputTarget(v) for v in range(19)], aug_smooth=False)
        #print(len(grayscale_cams),grayscale_cams[0].shape)
        # channels:['red', 'green', 'blue', 'yellow'] or MT, protein, nu, ER
        # cv2 is in format BGR
        #grayscale_cams = cam(input_tensor=inp, targets=[ClassifierOutputTarget(v) for v in range(19)], aug_smooth=True)
        bgr_img = rgb_img[...,[2,1,0]]
        _,bgreen = cv2.threshold(bgr_img[:,:,1], 20, 255, cv2.THRESH_BINARY)
        for v,k in LABEL_TO_ALIAS.items():
            prob = np.round(probs[v],2)
            grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(v)], aug_smooth=True).squeeze()
            #grayscale_cam = grayscale_cams[v]
            grayscale_cam = (grayscale_cam*255).astype('uint8')
            m = mse(bgr_img[:,:,1], grayscale_cam)
            s = ssim(bgr_img[:,:,1], grayscale_cam, mode='valid')
            
            # converting to its binary form cv2.threshold(img, thres1, thres2, thres_type)
            _,bcam = cv2.threshold(grayscale_cam, 10, 255, cv2.THRESH_BINARY)
            bs = ssim(bgreen, bcam, mode='valid')
            bm = mse(bgreen, bcam)
            iou_all = iou(bcam.flatten(), bgreen.flatten())
            iou_masked = iou(bcam.flatten()[compare_region], bgreen.flatten()[compare_region])

            #Pearson r of only masked pixels
            pearson_masked,pval_masked = pearsonr(bcam.flatten()[compare_region], bgreen.flatten()[compare_region])
            pearson,pearson_pval = pearsonr(bcam.flatten(), bgreen.flatten())

            sidebyside = np.r_[np.c_[bgr_img[:,:,1], grayscale_cam],np.c_[bgreen,bcam]]
            cv2.imwrite(f"{args.output_dir}/{iter_data['ID'][0]}_{iter_data['maskid'][0]}_{k}_{prob:.2f}.png", sidebyside)
            line = f"{iter_data['ID'][0]},{iter_data['maskid'][0]},{k},{prob},{s},{m},{bs},{bm},{pearson_masked},{pval_masked},{pearson},{pearson_pval},{iou_all},{iou_masked}\n"
            F.write(line)
    F.close()

def calculate_iou(args, test_loader):
    if os.path.exists(f"{args.output_dir}/result_df_singlelabel.csv"):
        df_singlelabel = pd.read_csv(f"{args.output_dir}/result_df_singlelabel.csv")
    else:
        labels_test = pd.read_csv(f'{DIR_CFGS.DATA_DIR}/raw/train.csv')
        labels_test = labels_test[labels_test.Label.isin(list([str(f) for f in range(19)]))] #Filtered for single label
        labels_test['LabelName'] = [LABEL_TO_ALIAS[int(f)] for f in labels_test.Label]
        
        if os.path.exists(f"{args.output_dir}/result_df_all_fixedcolumns.csv"):
            df = pd.read_csv(f"{args.output_dir}/result_df_all_fixedcolumns.csv")
        else:
            df = pd.read_csv(f"{args.output_dir}/result_df_all.csv")
            df.columns = ['label', 'prob', 'ssim', 'mse', 'binarized_ssim', 'binarized_mse', 'pearsonr_masked', 'pval_masked', 'pearsonr', 'pval', 'iou_all','iou_masked']
            df['image_id'] = [x[0] for x in df.index]
            df['cell_id'] = [x[1] for x in df.index]
            df['ID'] = ['_'.join([x[0], str(x[1])]) for x in df.index]
            df.to_csv(f"{args.output_dir}/result_df_all_fixedcolumns.csv", index=False)
        df_singlelabel = df.merge(labels_test, how='inner', left_on=['ID','label'], right_on=['ID','LabelName'])
    print(f'Number of single cells with single labels : {df_singlelabel.shape[0]}')
    #l_done = glob.glob(f'{DIR_CFGS.DATA_DIR}/1st_cams/tmp/*.png')
    #l_done = [l.rsplit('/',1)[1].rsplit('_',1)[0] for l in l_done]
    l_done = df_singlelabel[~df_singlelabel.iou_all.isin([0,1])].ID.to_list()
    print(f"Number of cells done {len(l_done)}")
    test_loader = generate_dataloader(args) if test_loader is None else test_loader
    for it, iter_data in tqdm(enumerate(test_loader, 0), total=len(test_loader), desc=f'cell default'):
        cell_id = "_".join((str(iter_data['ID'][0]),str(iter_data['maskid'][0])))
        if cell_id in l_done:
            continue

        compare_region = np.where(np.sum(iter_data['image'][0].numpy(), axis=0).flatten()>0)[0]

        i = df_singlelabel[df_singlelabel.ID==cell_id].index[0]
        r = df_singlelabel.iloc[i,]
        merged_img = plt.imread(glob.glob(f'{args.output_dir}/{r.image_id}_{r.cell_id}_{r.label}_*')[0])
        cam = merged_img[:128, 128:]
        green = merged_img[:128,:128]
        bgreen = merged_img[128:,:128]
        _,bcam = cv2.threshold(cam*255, 30, 1, cv2.THRESH_BINARY)
        
        overlap = bcam.astype('bool')*bgreen.astype('bool')
        union = bcam.astype('bool') + bgreen.astype('bool')
        iou_a = overlap.sum()/float(union.sum())
        df_singlelabel.loc[i,'iou_all'] = iou_a

        overlap = bcam.flatten()[compare_region].astype('bool') * bgreen.flatten()[compare_region].astype('bool')
        union = bcam.flatten()[compare_region].astype('bool') + bgreen.flatten()[compare_region].astype('bool')
        iou_m = overlap.sum()/float(union.sum())
        df_singlelabel.loc[i,'iou_masked'] = iou_m
        fig,ax = plt.subplots(2,3, sharex='all',sharey=True)
        ax[0,0].imshow(green)
        ax[0,1].imshow(cam)
        ax[1,0].imshow(bgreen)
        ax[1,0].text(10,10,s=f'IOU_a: {iou_a}',c='white')
        ax[1,1].imshow(bcam)
        ax[1,1].text(10,10,s=f'IOU_m: {iou_m}',c='white')
        ax[0,2].imshow(np.sum(iter_data['image'][0].numpy(), axis=0)>0)
        plt.savefig(f'{DIR_CFGS.DATA_DIR}/1st_cams/tmp/{r.image_id}_{r.cell_id}_{r.label}.png')
        if it % 1000 == 0:
            df_singlelabel.to_csv(f"{args.output_dir}/result_df_singlelabel1.csv", index=False)
    df_singlelabel.to_csv(f"{args.output_dir}/result_df_singlelabel1.csv", index=False)

def main(args):
    start_time = timer()
    args = initialize_environment(args)
    model = None
    test_loader = None
    #generate_cam(args, test_loader, model)
    #generate_cams_sclabel(args, test_loader, model)
    calculate_iou(args, test_loader)
    end_time = timer()
    print(f'time: {(end_time - start_time) / 60.:.2f} min.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--module', type=str, default='cls_efficientnet', help='model')
    parser.add_argument('--model_name', type=str, default='cls_efficientnet_b3', help='model_name')
    parser.add_argument('--model_dir', type=str, default=None, help='model_dir')
    parser.add_argument('--model_epoch', type=str, default='99.00_ema', help='model_epoch')
    parser.add_argument('--gpus', default=None, type=str, help='use gpus')
    parser.add_argument('--image_size', default='512', type=str, help='image_size')
    parser.add_argument('--batch_size', default=2, type=int, help='batch_size')
    parser.add_argument('--dataset', default='valid', type=str, help='dataset')
    parser.add_argument('--fold', default=0, type=int, help='index of fold')
    parser.add_argument('--augments', default='default', type=str, help='augments')
    parser.add_argument('--part_start', default=0., type=float, help='part_start')
    parser.add_argument('--part_end', default=1., type=float, help='part_end')
    parser.add_argument('--overwrite', default=0, type=int, help='overwrite')
    parser.add_argument('--ml_num_classes', default=ML_NUM_CLASSES, type=int)
    parser.add_argument('--label', type=int, default=None)
    parser.add_argument('--num_classes', default=NUM_CLASSES, type=int)
    parser.add_argument('--cell_type', type=int, default=0)
    parser.add_argument('--in_channels', type=int, default=4)
    parser.add_argument('--kaggle', type=int, default=0)
    parser.add_argument('--cell_complete', type=int, default=0)
    args = parser.parse_args()
    main(args)
