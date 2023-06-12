[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11103468&assignment_repo_type=AssignmentRepo)
## Technologies we use

<img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" /> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
<img src="https://img.shields.io/badge/PyCharm-000000.svg?&style=for-the-badge&logo=PyCharm&logoColor=white" /> <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" /> 


# Dog Breed Classifier
The Dog Breed Classifier project is a cutting-edge initiative that seeks to harness the power of artificial intelligence in the field of animal identification. By leveraging the latest advancements in machine learning and image recognition, this project aims to identify up to 120 distinct dog breeds through digital images. With an expansive database encompassing the unique physical characteristics of each breed, our AI-powered tool is designed to enhance our understanding of these beloved pets, and aid in their identification.

The primary objective of the Dog Breed Classifier project is to develop an advanced machine learning model capable of identifying up to 120 different dog breeds from images. This will involve rigorous training of the model using a vast and diverse dataset to ensure high levels of accuracy and precision, considering the wide variations in size, color, and angle of image capture across different dog breeds. But this project isn't just about the creation of an AI model, it extends to the development of an user-friendly web application that allows users to upload images of dogs and receive instant information about their breed.

## Code structure
Our code follows the next structure :
```bash
├── data
│   ├── FolderClassification.py
│   ├── BackgroundRemove.py
│   ├── datos.txt
│   └── ImageExamples.py
├── models
│   ├── __init__.py
│   └── model.py
├── utils
│   ├── __init__.py
│   ├── DataLoader.py
│   ├── Transformations.py
│   ├── FeatureExtraction.py
│   ├── heatmap.py
│   ├── train.py
│   ├── test.py
│   ├── menu.py
│   └── utils.py
└── Dog-Breed-classif
│   ├── test
│   │   ├── affenpinscher
│   │   │   └── ...jpg
│   │   ├── afghan_hound
│   │   |   └── ..jpg
│   |   └── ...
│   ├── train
│   │   ├── affenpinscher
│   │   │   └── ...jpg
│   │   ├── afghan_hound
│   │   |   └── ..jpg
│   |   └── ...
│   └── labels.csv
└── main.py
├── README.md
├── LICENSE
├── enviroment.yml
├── .gitignore
├── Requirements.txt
└── Demostracio_resultats.ipynb
```

## Code
The given code allows us to run the training of the data and its test with several models with the Dog Breed dataset. To run it, we need tho download the github. Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml] file has all the required dependencies. 

Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it, this will take a while:


```
conda env create -f environment.yml
```

```
conda activate DogBreed
```
### Before the main code
You have to download the kaggle folders and unzip them on a folder called `Dog-Breed-classif` so you have the following estructure:
```bash
Dog-Breed-classif
  ├── test
  ├── train
  └── labels.csv
```

Then you have to execute the `FolderClassification.py` script on the data folder. After this the code is ready to be executed.

To run the **main** code:
```
python main.py
```

A menu will be displayed at the terminal where the user can select which actions to take. It is recommended that the first time it is used a model is trained to be able to take the test later. The menu will also give us the option to save the weights at the end of the training and save the graphs of the metrics to have acces at any desired moment to the results of the training. The test option will take the selected weights saved from a previous training and show us the accuracy obtained with the test dataset part. Finally, we are also given the option of train and test which brings the two things together.

### Considerations to have:

The weight loading in the only test option of the menu needs to open the saved weigths of the selected model in the folder `.\models`. So if the original model is a pretrained one we have to select "y" when the menu asks us if we want it to be pretrained, the feature extraction option has no impact because then we will load our saved weights. Also in the part of the optimizer it is only to take the optimizer's name so the different parameters selected during this specific part of the menu will have no impact on the test we will do to our model. This redundant option selection it is due a lack of time.

In addition, we performed several tests removing the background in the dataset without success. Since we have few images for training and they could not be used to train with pre-trained models because these have been trained with complete images and passing these new ones did not improve the accuracy. Consequently, we have mentioned it in the work but we have not used it. It should be added that the library that allows us to remove the background has compatibility problems with some more updated versions of other libraries, we recommend using Google Colab as it has all the compatibilities.

## Contributors
Here are the names and UAB mails of each group member:

- Sofia Di Capua, sofiadicapua29@gmail.com
- Biel González, agirzel@gmail.com
- Cristina Soler, csolerare@gmail.com

| [<img src="https://avatars.githubusercontent.com/u/73697639?v=4" width=115><br><sub>Sofia Di Capua</sub>](https://github.com/SofiaDiCapua) |  [<img src="https://avatars.githubusercontent.com/u/81986384?v=4" width=115><br><sub>Biel González</sub>](https://github.com/Zynokrex) |  [<img src="https://avatars.githubusercontent.com/u/58566857?v=4" width=115><br><sub>Cristina Soler</sub>](https://github.com/kermitsc7) |
| :---: | :---: | :---: |


Xarxes Neuronals i Aprenentatge Profund
__Computational Mathematics & Data analyitics__ degree, UAB, 2023
