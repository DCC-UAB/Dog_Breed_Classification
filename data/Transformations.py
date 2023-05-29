import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os
import shutil
import torch
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder


import warnings
warnings.filterwarnings('ignore')

#Loading example image
path="..\Dog-Breed-classif"
img_example=Image.open("test/00a3edd22dc7859c487a64777fc8d093.jpg")

#Global variables
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

input_size = 224

def Show_Image(Image, Picture_Name):
    plt.imshow(Image)
    plt.title(Picture_Name)
    
#Transformation 1 (Basic)
data_transforms1 = {
    'train': transforms.Compose([
        transforms.Resize(input_size), #Establim la mida a 224
        transforms.CenterCrop(input_size), #Centrem el tallat de les imatges
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
}

#Watch the transformation1
data_transforms1 = transforms.Compose([
        transforms.Resize(input_size), #Establim la mida a 224
        transforms.CenterCrop(input_size), #Centrem el tallat de les imatges
        transforms.ToTensor(),
        normalize
    ])
T1 = data_transforms1(img_example)
Show_Image(T1.permute(1, 2, 0), 'Trasformation1 Image')

#Transformation 2 (Grey Scale)
data_transforms2 = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.Grayscale(3), #Output channels 3, RGB but in gray scale
        transforms.ToTensor()

    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
}

#Watch the transformation 2
data_transforms2 = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.Grayscale(3), #Output channels 3, RGB but in gray scale
        transforms.ToTensor()
    ])

T2 = data_transforms2(img_example)
Show_Image(T2.permute(1, 2, 0), 'Trasformation2 Image')

#Transformation 3 (Complete one)
data_transforms3 = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomVerticalFlip(p=0.5), #Girem verticalment
        transforms.RandomHorizontalFlip(p=0.5), #Girem horitzontalment
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), #Apliquem un blur
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
}

#Watch the transformation 3
data_transforms3 = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomVerticalFlip(p=0.5), #Girem verticalment
        transforms.RandomHorizontalFlip(p=0.6), #Girem horitzontalment
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), #Apliquem un blur
        transforms.ToTensor(),
        normalize
    ])
T3 = data_transforms3(img_example)
Show_Image(T3.permute(1, 2, 0), 'Trasformation3 Image')

#Transformation 4 (jitter)
data_transforms4 = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ColorJitter(brightness=.5, hue=.3), #Jitter color
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
}

#Watch the transformation 4
data_transforms4= transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ColorJitter(brightness=.5, hue=.3), #Jitter color
        transforms.ToTensor()
    ])
T4 = data_transforms4(img_example)
Show_Image(T4.permute(1, 2, 0), 'Trasformation4 Image')
