import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import os
import shutil
import torch
import torchvision.transforms as T

import warnings
warnings.filterwarnings('ignore')

#Preparing variables
n=4

#Loading example image
path="..\Dog-Breed-classif"
img_example=Image.open(path+"/test/00a3edd22dc7859c487a64777fc8d093.jpg")

#Showing original image
plt.title('Original Image')
plt.imshow(img_example)
plt.show()

#Showing grayscale image
grayscale_fig=plt.figure()
plt.title('Grayscale image')
gray_img=ImageOps.grayscale(ImageOps.invert(img_example))
plt.imshow(gray_img, cmap="Greys")
plt.show()

#Showing normalized image
normalized_fig=plt.figure()
plt.title('Normalized image')
prepared_img = T.ToTensor()(img_example)
mean = prepared_img.mean([1, 2])
std = prepared_img.std([1, 2])
normalize = T.Normalize(mean, std)
normalized_img = normalize(prepared_img)
plt.imshow(normalized_img.permute(1, 2, 0))
plt.show()

#Showing jitted images
jitted_fig, axs = plt.subplots(2, 2)
ax=axs.flatten()
jitter=T.ColorJitter(brightness=.5, hue=.3)
jitted_imgs=[jitter(img_example) for i in range(n)]
for i in range(n): ax[i].imshow(jitted_imgs[i])
jitted_fig.suptitle('Jitted images')
plt.show()

#Showing blurred images
blurred_fig, axs = plt.subplots(2, 2)
ax=axs.flatten()
blurrer=T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
blurred_imgs=[blurrer(img_example) for i in range(n)]
for i in range(n): ax[i].imshow(blurred_imgs[i])
blurred_fig.suptitle('blurred images')
plt.show()

#Showing random horizontal flipped images
h_flip_fig, axs = plt.subplots(2, 2)
ax=axs.flatten()
h_flipper=T.RandomHorizontalFlip(p=0.5)
h_flipped_imgs=[h_flipper(img_example) for i in range(n)]
for i in range(n): ax[i].imshow(h_flipped_imgs[i])
h_flip_fig.suptitle('Horizontal flipped images')
plt.show()

#Showing random vertical flipped images
v_flip_fig, axs = plt.subplots(2, 2)
ax=axs.flatten()
v_flipper=T.RandomVerticalFlip(p=0.5)
v_flipped_imgs=[v_flipper(img_example) for i in range(n)]
for i in range(n): ax[i].imshow(v_flipped_imgs[i])
v_flip_fig.suptitle('Vertical flipped images')
plt.show()

#Combining some transformations
transformer = T.Compose([jitter, blurrer, h_flipper, v_flipper])
comb_fig, axs = plt.subplots(2, 2)
ax=axs.flatten()
combined_transform_imgs=[transformer(img_example) for i in range(n)]
for i in range(n): ax[i].imshow(combined_transform_imgs[i])
comb_fig.suptitle('Combined transformations on an image')
plt.show()

#Saving images
print("Do you want to save the generated images?[y/n]\n")
saving=input()
if(saving=="y"):
  galery_path=os.getcwd()+"\Image_galery"
  if os.path.exists(galery_path): shutil.rmtree(galery_path)
  os.mkdir(galery_path)
  print("Image galery created")
  grayscale_fig.savefig(".\Image_galery\Grayscale_dog.png")
  normalized_fig.savefig(".\Image_galery/Normalized_dog.png")
  jitted_fig.savefig(".\Image_galery\Jitted_dog.png")
  blurred_fig.savefig(".\Image_galery\Blurred_dog.png")
  h_flip_fig.savefig(".\Image_galery\H_flipped_dog.png")
  v_flip_fig.savefig(".\Image_galery\V_flipped_dog.png")
  comb_fig.savefig(".\Image_galery\Combined_transforms_dog.png")
  print("Images saved succesfully")

