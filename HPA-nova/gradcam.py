from xml.etree.ElementInclude import include
import numpy as np
import cv2
from skimage.io import imread
import tensorflow as tf
from PIL import Image
import torch.nn as nn
import torch
import torchvision
from torchvision.transforms.transforms import ToPILImage 
from torchvision import transforms
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torch.autograd import Variable

class novaGradCAM:

    def __init__(self, model, img_path, target_size):
        super(novaGradCAM,self).__init__()
        self.img_path = img_path
        self.target_size = target_size
        self.gradients = None # placeholder for the gradients
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.model.eval()
        # Set a specific layer as trainable
        # print(self.model)
        # self.target_layers = [self.model.fc2] # for alexnet
        self.target_layers = [self.model.net.layer4] # for resnet50
        # self.target_layers = [self.model.model.blocks[-1]] # for tf_efficientnet_b3 (pretrained)
        # self.target_layers = [self.model.extractor[-1]] # for efficientnet2 (newly trained)
        # self.target_layers = [self.model.blocks[-1].norm1] # for vit (newly trained)
        # self.target_layers = [self.model.mlp_cells]

        # print(f'MODEL:{self.model}')
        # print(self.model.model.blocks[-1][1].bn3)
        # self.target_layers = [self.model.model.blocks[-1][1]] 
            
        # print(f'self.target_layers: {self.target_layers}')

    
    def load(self):
        image_titles = [self.img_path]
        img = imread(self.img_path)
        img = img[:,:,::-1]
        img=np.ascontiguousarray(img)
        img = cv2.resize(img, (self.target_size, self.target_size))
        image_prep = preprocess_image(img, mean=[0.0979, 0.06449, 0.062307, 0.098419], std=[0.14823, 0.0993746, 0.161757, 0.144149])#.requires_grad_(True)
        img = np.transpose(img, (2,0,1))
        img = np.float32(img)/255
        
        return img, image_prep # img: (4, 256, 256), image_prep: torch.Size([1, 4, 256, 256])

    
    # Create a Gradcam object
    def make_gradcam(self, image, image_prep, folder_path, vit=False):

        with torch.enable_grad():
            cam = GradCAM(model=self.model, target_layers=self.target_layers,use_cuda=True)
            for i in range(19):
                # grayscale_cam = cam(input_tensor=image.to(self.device).requires_grad_(True), targets=None)
                if not vit:
                    targets = [ClassifierOutputTarget(i)]
                else:
                    for param in self.model.blocks.parameters():
                        param.requires_grad = True
                    # image_prep = image_prep.squeeze(0)
                    targets = None
                print(f'targets: {targets}, image_prep: {image_prep.shape}')
                grayscale_cam = cam(input_tensor=image_prep.to(self.device), targets=targets)
                grayscale_cam = grayscale_cam[0]
                img = np.transpose(image, (1,2,0))
                # visualization = show_cam_on_image(img, grayscale_cam) # in show_cam_on_image: 'cam = (1 - image_weight) * heatmap + image_weight * img'
                # cv2.imwrite(f'results/{folder_path}/logs/class{i}.jpg', visualization)

                image_weight = 0.6
                heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap = np.float32(heatmap) / 255
            
                if heatmap.shape[-1] < img.shape[-1]:
                    # Pad the heatmap array with zeros to match the number of channels in img
                    pad_channels = img.shape[-1] - heatmap.shape[-1]
                    heatmap = np.pad(heatmap, ((0, 0), (0, 0), (0, pad_channels)), mode='constant')
            
                # Ensure both arrays have the same shape
                if heatmap.shape != img.shape:
                    raise ValueError("Heatmap and image must have the same shape")
                visualization = (1 - image_weight) * heatmap + image_weight * img
                visualization = visualization / np.max(visualization)
                visualization = np.uint8(255 * visualization)

                # print(f'visualization: {visualization.shape}') # (256, 256, 4)
                cv2.imwrite(f'results/{folder_path}/logs/class{i}.jpg', visualization)
                            
                

                





