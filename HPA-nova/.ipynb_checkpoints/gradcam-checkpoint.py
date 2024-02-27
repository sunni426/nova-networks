from xml.etree.ElementInclude import include
import numpy as np
import cv2
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
        # print(self.model)
        self.model.eval()
        self.target_layers = [self.model.net.layer4]
        # print(f'self.target_layers {self.target_layers}')
        # self.cam = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=(True))
        
    def load(self):
        image_titles = [self.img_path]
        img = cv2.imread(self.img_path, 1)
        img = img[:,:,::-1]
        img=np.ascontiguousarray(img)
        img = cv2.resize(img, (self.target_size,self.target_size))
        img_array = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        img_array = np.float32(img_array)/255
         
        image_prep = preprocess_image(img_array, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#.requires_grad_(True)  
        return img_array, image_prep 

    # Create a Gradcam object
    def make_gradcam(self, image, image_prep, folder_path):

        with torch.enable_grad():
            cam = GradCAM(model=self.model, target_layers=self.target_layers,use_cuda=True)# use_cuda=('cuda' == self.device))
            # print(f'cam {cam}')
            for i in range(19):
                # grayscale_cam = cam(input_tensor=image.to(self.device).requires_grad_(True), targets=None)
                targets = [ClassifierOutputTarget(i)]
                grayscale_cam = cam(input_tensor=image_prep.to(self.device), targets=targets)
                # print(f'image: {image.shape}, grayscale_cam {grayscale_cam.shape}')
                grayscale_cam = grayscale_cam[0]
                visualization = show_cam_on_image(image, grayscale_cam)
                cv2.imwrite(f'results/{folder_path}/logs/class{i}.jpg', visualization)
