from rembg import remove
from PIL import Image
import os
import sys
import pandas as pd

"""
Aquest codi només funciona al Google Colaboratory i donat que les imatges sense fons resulten en un pitjor entrenament no cal executar aquest codi
"""

labels = pd.read_csv("../Dog-Breed-classif/labels.csv")

#Set the input and outout directories, we need to create a folder in the directory to save the data
dirc = '../Dog-Breed-classif/train'
dircout = '../Dog-Breed-classif/trainBR'

#Get the list of the file names in the input directory
files_names = os.listdir(dirc)

#Get the number of images to process from user input
N = int(input("Select a number (int) of images:"))

#Check if the number is bigger than the num of images
if(len(files_names) < N):
    sys.exit()


for file_name in files_names[:N]:
    #Set the input and the output directory of the images
    input_path = dirc + '/' + file_name
    output_path = dircout + '/' + file_name
    
    #Open the image and convert to RGB format
    input = Image.open(input_path).convert('RGB')
    input.save(file_name)
    
    #Remove the background using remgb library
    output = remove(input)
    
    #Save the output image with transparent background in PNG format
    output.save(output_path, format ='PNG')
    

list_names = []
df = pd.DataFrame(columns=['id', 'breed'])

#Save all the names of the files without the image extension (last 4 characters)
for file_name in files_names:
  list_names.append(file_name[:-4])


for ID in list_names:
  if(ID in labels.id.values):
    # Get the corresponding breed from the 'breed' column
    b = labels[labels.id == ID].breed.values[0]
    # Append the image name and breed as a new row in the DataFrame
    df = df.append({'id': ID, 'breed':b}, ignore_index=True)


df.to_csv("Dog-Breed-classif/labelsB.csv")
  
