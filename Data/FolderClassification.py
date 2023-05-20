import os
import sys
import shutil
import pandas as pd

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
    shutil.move(path_img, new_path)

print("Images moved")
