# pip install --upgrade tf-keras-vis tensorflow
# =============================================

from xml.etree.ElementInclude import include
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import torch.nn as nn

from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

# define the grand cam function
from tf_keras_vis.utils import normalize
from matplotlib import cm 
from tf_keras_vis.gradcam import Gradcam

# converting pytorch to TF
from pytorch2keras import pytorch_to_keras

class GradCAM:

    def __init__(self, model, img_path, target_size):
        super(GradCAM,self).__init__()
        # self.model = model
        self.img_path = img_path
        self.target_size = target_size

        # Convert model to keras (for TF)
        input_np = np.random.uniform(0, 1, (1, 10, target_size, target_size)) # dummy variable for shape
        input_var = Variable(torch.FloatTensor(input_np))
        self.model = pytorch_to_keras(model, input_var, [(10, None, None)], verbose=True)  
        
    
    def load(self):
        #load the Vgg16 model
        model = self.model
        # print(model.summary())
        
        image_titles = [self.img_path]
        # load the images
        img1=load_img(self.img_path, target_size=(self.target_size,self.target_size))
        print(type(img1))
        # convert it to Numpy Array:
        img1_array = cv2.cvtColor(np.array(img1),cv2.COLOR_RGB2BGR)
        
        # Show the images
        cv2.imwrite(f"Original_{self.img_path}", img1_array)
        
        # prepare the data for the Vgg16 model
        # images = np.asarray([np.array(img1), np.array(img2)])
        images = img1_array
        print(f'images: {images.shape}')
        X = preprocess_input(images)
        
        return img1_array, X

    # define the loss functions with a traget classes 
    def loss(self,output):
        return(output[0][283], output[1][150])
    
    # define the model modifier - change the activation function
    def model_modifier(self,mdl):
        # mdl.layers[-1].activation = tf.keras.activations.linear # we change the activation function of last layer to linear
        # For our PyTorch model
        # mdl[-1].activation = nn.Identity() # nn.Identity() is used as a replacement for a linear activation function because it effectively performs a linear transformation without any activation
        mdl = mdl
        
    # Create a Gradcam object
    def make_gradcam(self, X):
        gradcam = Gradcam(self.model,
                        model_modifier=self.model_modifier,
                        clone=False)
        cam = gradcam(self.loss, X, penultimate_layer=-1 )# the layer befor the softmax
        cam = normalize(cam)
        return cam
    
    # Visualize outcome
    def visualize(self, img1_array, cam):
    # to extract the image from the model 
        heatmapImg1 = np.uint8(cm.jet(cam[0])[..., :3] * 255 )
        # chnage the color map to jet
        heatmapImg1 = cv2.applyColorMap(heatmapImg1 , cv2.COLORMAP_JET)
        
        # lets add some alpha transparency
        alpha = 0.5
        overlay = heatmapImg1.copy() # copy the image
        result1 = cv2.addWeighted(img1_array, alpha, heatmapImg1 , 1-alpha, 0)
        
        scale_precent = 200
        w = int(heatmapImg1.shape[1] * scale_precent / 100)
        h = int(heatmapImg1.shape[0] * scale_precent / 100)
        dim = (w,h)
        
        result1 = cv2.resize(result1, dim , interpolation=cv2.INTER_AREA)
        img1_array = cv2.resize(img1_array, dim, interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(f"GradCam{self.img_path}.jpg",result1 )


        