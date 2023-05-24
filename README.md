[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11103468&assignment_repo_type=AssignmentRepo)
# XNAP-Project title (replace it by the title of your project)
Write here a short summary about your project. The text must include a short introduction and the targeted goals

The Dog Breed Classifier project is a cutting-edge initiative that seeks to harness the power of artificial intelligence in the field of animal identification. By leveraging the latest advancements in machine learning and image recognition, this project aims to identify up to 120 distinct dog breeds through digital images. With an expansive database encompassing the unique physical characteristics of each breed, our AI-powered tool is designed to enhance our understanding of these beloved pets, and aid in their identification.

The primary objective of the Dog Breed Classifier project is to develop an advanced machine learning model capable of identifying up to 120 different dog breeds from images. This will involve rigorous training of the model using a vast and diverse dataset to ensure high levels of accuracy and precision, considering the wide variations in size, color, and angle of image capture across different dog breeds. But this project isn't just about the creation of an AI model, it extends to the development of an user-friendly web application that allows users to upload images of dogs and receive instant information about their breed.

## Code structure
Our code follows the next structure :
```bash
├── Data
│   ├── __init__.py
│   ├── FolderClassification.py
│   └── ImageExamples.py
├── models
│   ├── __init__.py
│   ├── model1.py
│   ├── model2.py
│   └── model3.py
├── utils
│   ├── __init__.py
│   ├── DataLoader.py
│   ├── Transformations.py
│   └── utils.py
└── Dog-Breed-classif
    ├── test
    │   ├── affenpinscher
    │   │   └── ...jpg
    │   ├── afghan_hound
    │   |   └── ..jpg
    |   └── ...
    ├── train
    │   ├── affenpinscher
    │   │   └── ...jpg
    │   ├── afghan_hound
    │   |   └── ..jpg
    |   └── ...
    ├── labels.csv
    └── sample_submission.csv
```
## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```

## Contributors
Here are the names and UAB mails of each group member:
- Biel González, agirzel@gmail.com
- Cristina Soler, csolerare@gmail.com
- Sofia Di Capua, sofiadicapua29@gmail.com


Xarxes Neuronals i Aprenentatge Profund
__Computational Mathematics & Data analyitics__ degree, UAB, 2023
