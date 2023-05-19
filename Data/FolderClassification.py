import os
import sys
import shutil
import pandas as pd

path_labels = '../Dog-Breed-classif/labels.csv'
path_imgs_train = '../Dog-Breed-classif/train'

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
    shutil.move(path_img, new_path)

print("Images moved")
