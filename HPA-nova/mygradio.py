import gradio as gr
import torch
from configs import Config
from PIL import Image
import cv2
from utils import parse_args, prepare_for_result
from dataloaders import get_dataloader
from models import get_model
from losses import get_loss, get_class_balanced_weighted
from losses.regular import class_balanced_ce
from functools import partial
from optimizers import get_optimizer
from basic_train import basic_train, basic_test, mean_average_precision
from scheduler import get_scheduler
from utils import load_matched_state
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip, RandomAffine, RandomVerticalFlip, RandomChoice, ColorJitter, RandomRotation)
from basic_train import tta_validate
from torchvision import transforms
import torch
try:
    from apex import amp
except:
    pass
import albumentations as A
from dataloaders.transform_loader import get_tfms
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from skimage.io import imread

@torch.inference_mode()
def get_best_model():
    args, cfg = parse_args()
    if args.mode == 'gradio':
        
        df = pd.read_csv(
            'results/' + cfg.basic.id + '/train.log', sep='\t')
        if args.epoch > 0:
            best_epoch = args.epoch
        else:
            if 'loss' in args.select:
                asc = True
            else:
                asc = False
            best_epoch = int(df.sort_values(args.select, ascending=asc).iloc[0].Epochs)
        print('Best check we use is: {}'.format('f{}_epoch-{}.pth'.format(cfg.experiment.run_fold, best_epoch)))
        
        test_dl = get_dataloader(cfg)(cfg).get_dataloader(test_only=True, tta=args.tta, tta_tfms=None)
        _, valid_dl, _ = get_dataloader(cfg)(cfg).get_dataloader(tta=args.tta, tta_tfms=None)
        # loading model
        model = get_model(cfg)
        model.load_state_dict(torch.load(
            Path(os.path.dirname(os.path.realpath(__file__))) / 'results' / cfg.basic.id / 'checkpoints' / 'f0_epoch{}.pth'.format(cfg.experiment.run_fold, best_epoch),
            map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu', 'cuda:2': 'cpu', 'cuda:3': 'cpu'}
        ))
        model = model.cpu()
        if len(cfg.basic.GPU) == 1:
            print('[ W ] single gpu prediction the gpus is {}'.format(cfg.basic.GPU))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # torch.cuda.set_device(int(cfg.basic.GPU))
            if device == 'cuda':
                torch.cuda.set_device(0)
            model = model.cuda()
        else:
            print('[ W ] dp prediction the gpus is {}'.format(cfg.basic.GPU))
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=[int(x) for x in cfg.basic.GPU])

    return model
    
def predict(input_image, threshold=0.5, model=None, preprocess_fn=None, device="cuda", idx2labels=None, cfg=Config):
    input = np.array(input_image)
    R = input[:, :, 0]
    G = input[:, :, 1]
    B = input[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B # Calculate luminance using the formula
    rgby_image = np.dstack((R, G, B, Y))
    
    rgby_input = np.transpose(rgby_image, (2,0,1))
    print(f'{rgby_input.shape}')
    input_tensor = torch.tensor(rgby_input)
    input_tensor = input_tensor.unsqueeze(0).to(device).float()
    print(f'{input_tensor.shape}')
    model.eval()
    with torch.no_grad():
        model = get_best_model()
        model.eval()
        # Generate predictions
        output = model(input_tensor)
        output = torch.stack(output).cuda().detach()
        
        probabilities = torch.sigmoid(output)[0].cpu().numpy().tolist()
        probabilities = probabilities[0]
        
        output_probs = dict()
        predicted_classes = []
     
        for idx, prob in enumerate(probabilities):
            output_probs[idx2labels[idx]] = prob
            if prob >= threshold:
                predicted_classes.append(idx2labels[idx])
                print(f'predicted_classes: {predicted_classes}')
                
        predicted_classes = "\n".join(predicted_classes)
        print(f'predicted_classes:{predicted_classes}') 
        return predicted_classes, output_probs
 
 
if __name__ == "__main__":
    labels = {
        0: "Nucleoplasm",
        1: "Nuclear membrane",
        2: "Nucleoli",
        3: "Nucleoli fibrillar center",
        4: "Nuclear speckles",
        5: "Nuclear bodies",
        6: "Endoplasmic reticulum Cytosol",
        7: "Golgi apparatus",
        8: "Intermediate filaments",
        9: "Actin filaments",
        10: "Microtubules",
        11: "Mitotic spindle",
        12: "Centrosome",
        13: "Plasma membrane",
        14: "Mitochondria",
        15: "Aggresome",
        16: "Cytosol",
        17: "Vesicles and punctate cytosolic patterns",
        18: "Negative",
    }
 

    with torch.no_grad():
        DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        args, cfg = parse_args()
        model = get_best_model()
        model.to(DEVICE)
        model.eval()
    
        preprocess = Compose(
            [
                Resize(size=256),
                ToTensor(),
                Normalize(mean=[0.0979, 0.06449, 0.062307, 0.098419], std=[0.14823, 0.0993746, 0.161757, 0.144149],inplace=True),
            ]
        )
        
        images_dir = '../preprocessing/train_5000/cell_10'
        file_names = os.listdir(images_dir)
        examples = [[os.path.join(images_dir, file_name), 0.4] for file_name in np.random.choice(file_names, size=10, replace=False)]
        print(examples)
        
        # examples = [[i, 0.4] for i in np.random.choice(images_dir, size=10, replace=False)]

        iface = gr.Interface(
            fn=partial(predict, model=model, preprocess_fn=preprocess, device=DEVICE, idx2labels=labels),
            inputs=[
                gr.Image(type="pil", label="Image"),
                gr.Slider(0.0, 1.0, value=0.5, label="Threshold", info="Select the cut-off threshold for a node to be considered as a valid output."),
            ],
            outputs=[
                gr.Textbox(label="Labels Present"),
                gr.Label(label="Probabilities", show_label=False),
            ],
            examples=examples,
            cache_examples=False,
            allow_flagging="never",
            title="Medical Multi-Label Image Classification",
        )
     
        iface.launch() #public: share=True