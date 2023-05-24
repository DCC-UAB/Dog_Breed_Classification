# Imports
import os
import sys
import shutil
import pandas as pd

"""
The provided code functions to classify images into directories according to their breed. If a directory corresponding 
to a breed already exists, the code refrains from creating a new one; however, if there is no existing directory for a 
certain breed, it automatically generates one. 
"""

# Checking if we are in the correct directory, if don't please go where labels and train are accesible
print("Input the path to the dataset images (Current working directory:", os.getcwd(), ")")
path = input()
path_labels = path+'/labels.csv'
path_imgs_train = path+'/train'

train_labels = pd.read_csv(path_labels)
unique_labels = train_labels["breed"].unique()

print("Creating folders...")
for label in unique_labels:
    try:
        os.mkdir(path_imgs_train+"/"+label)
    except:
        print("Folder already exists, exiting execution...")
        sys.exit()

print("Moving images...")
for name, label in zip(train_labels.id, train_labels.breed):
    path_img = path_imgs_train+"/"+name+".jpg"
    new_path = path_imgs_train + "/" + label + "/" + name + ".jpg"
    try:
        shutil.move(path_img, new_path)
    except:
        print("Image not found, proceeding with the rest")

print("Images moved")
