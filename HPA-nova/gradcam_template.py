# pip install --upgrade tf-keras-vis tensorflow
# =============================================

from xml.etree.ElementInclude import include
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img


#load the Vgg16 model
model = Model(weights='imagenet',include_top=True)
print(model.summary())

# here is the link for imagenet classes : https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/

image_titles = ['Persian-cat','Sea-lion']

# load the images

img1=load_img("Persian-cat.jpg", target_size=(224,224))
img2=load_img("Sea-lion.jpg", target_size=(224,224))

print(type(img1))

# convert it to Numpy Array:
img1_array = cv2.cvtColor(np.array(img1),cv2.COLOR_RGB2BGR)
img2_array = cv2.cvtColor(np.array(img2),cv2.COLOR_RGB2BGR)

# Show the images 

# cv2.imshow("Original - Cat ", img1_array)
# cv2.imshow("Original - Sea-lion", img2_array)
cv2.imwrite("Original-Cat.png", img1_array)
cv2.imwrite("Original-Sea-lion.png", img2_array)
# cv2.waitKey(0)

# prepare the data for the Vgg16 model
images = np.asarray([np.array(img1), np.array(img2)])
print(f'images: {images.shape}')
X = preprocess_input(images)

# define the loss functions with a traget classes 
def loss(output):
    return(output[0][283], output[1][150])


# define the model modifier - change the activation function
def model_modifier(mdl):
    mdl.layers[-1].activation = tf.keras.activations.linear # we change the activation function of last layer to linear


# define the grand cam function
from tf_keras_vis.utils import normalize
from matplotlib import cm 
from tf_keras_vis.gradcam import Gradcam

# create an object

gradcam = Gradcam(model,
                model_modifier=model_modifier,
                clone=False)


cam = gradcam(loss, X , penultimate_layer=-1 )# the layer befor the softmax

cam = normalize(cam)

# lets show the outcome :

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
img1_array = cv2.resize(img1_array, dim , interpolation=cv2.INTER_AREA)

# cv2.imshow("GradCam - Cat",result1 )
cv2.imwrite("GradCam - Cat.jpg",result1 )
# cv2.imshow("Original - Cat",img1_array )
# cv2.waitKey(0)

# lets show the sea lion

heatmapImg2 = np.uint8(cm.jet(cam[1])[..., :3] * 255 )
heatmapImg2 = cv2.applyColorMap(heatmapImg2 , cv2.COLORMAP_JET)
overlay = heatmapImg2.copy() # copy the image
result2 = cv2.addWeighted(img2_array, alpha, heatmapImg2 , 1-alpha, 0)

w = int(heatmapImg2.shape[1] * scale_precent / 100)
h = int(heatmapImg2.shape[0] * scale_precent / 100)
dim = (w,h)

result2 = cv2.resize(result2, dim , interpolation=cv2.INTER_AREA)
img2_array = cv2.resize(img2_array, dim , interpolation=cv2.INTER_AREA)

# cv2.imshow("GradCam - Sea lion",result2 )
# cv2.imshow("Original - Sea lion",img2_array )
cv2.imwrite("GradCam - Sea-lion.jpg",result2 )
# cv2.waitKey(0)


# # pip install torch torchvision
# # =============================================

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from torchvision import models, transforms
# from PIL import Image

# import torch
# from torch.autograd import Function
# from torchvision import models

# # Load the Vgg16 model
# model = models.vgg16(pretrained=True)
# model.eval()

# # 283 Persian cat, 150 sea lion
# target_classes = [283, 150]

# # Image titles
# image_titles = ['Persian-cat', 'Sea-lion']

# # Load the images
# img1 = Image.open("Persian-cat.jpg").convert("RGB")
# img2 = Image.open("Sea-lion.jpg").convert("RGB")

# # Convert images to Numpy Array
# img1_array = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
# img2_array = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

# # Show the images
# # cv2.imshow("Original - Cat", img1_array)
# # cv2.imshow("Original - Sea-lion", img2_array)
# cv2.imwrite("Original-Cat.png", img1_array)
# cv2.imwrite("Original-Sea-lion.png", img2_array)
# # cv2.waitKey(0)

# # Prepare the data for the Vgg16 model
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# images = [preprocess(img1), preprocess(img2)]
# X = torch.stack(images)

# # Define the loss functions with target classes
# def loss(output):
#     return output[0, target_classes[0]], output[1, target_classes[1]]

# # Define the model modifier - change the activation function
# def model_modifier(mdl):
#     mdl.features[-1].register_forward_hook(hook_fn)

# # Define the hook function for the last convolutional layer
# def hook_fn(module, input, output):
#     module.backward = output.clone()

# # Define the GradCam function
# class GradCam:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.model.eval()

#     def forward(self, x):
#         return self.model(x)

#     def __call__(self, x, index=None):
#         output = self.forward(x)

#         if index is None:
#             index = torch.argmax(output)

#         self.model.features[-1].backward = None  # Clear previous backward hook

#         # Calculate gradients
#         print(f'output: {output.shape}')
#         output.squeeze(0)[index].backward()
#         print(f'output: {output.shape}')

#         # Get gradients from the last convolutional layer
#         print(f'self.model.features[-1]: {self.model.features[-1]}')
#         gradients = self.model.features[-1].backward

#         print(f'gradients: {gradients}')
#         pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
#         activations = self.model.features[-1].forward(x)

#         # Weight the channels by the gradients
#         weighted_activations = activations * pooled_gradients

#         # Sum the channels
#         heatmap = torch.mean(weighted_activations, dim=1, keepdim=True)

#         # ReLU on the heatmap
#         heatmap = F.relu(heatmap)

#         # Normalize the heatmap
#         heatmap /= torch.max(heatmap)

#         return heatmap

# gradcam = GradCam(model, target_layer=-2)  # Use the last convolutional layer

# # Apply GradCam to each image
# for i, title in enumerate(image_titles):
#     heatmap = gradcam(X[i:i + 1], index=target_classes[i])

#     # Convert to numpy array and resize
#     heatmap_np = heatmap.detach().numpy()[0, 0]
#     heatmap_np = cv2.resize(heatmap_np, (img1_array.shape[1], img1_array.shape[0]))

#     # Apply colormap
#     heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_np), cv2.COLORMAP_JET)

#     # Overlay the heatmap on the image
#     result = cv2.addWeighted(img1_array if i == 0 else img2_array, 0.5, heatmap_colored, 0.5, 0)

#     # Display the result
#     # cv2.imshow(f"GradCam - {title}", result)
#     cv2.imwrite(f"GradCam - {title}", result)
#     # cv2.waitKey(0)
