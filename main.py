import os
import random
import torch

import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


from train import *
from utils.utils import model_metrics_plot
from utils.DataLoader import *
from utils.Transformations import *
from models.models import *
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training_pipeline(model_to_train, model_name, dataloaders, epochs=25,
                      learning_rate=0.001, momentum=0.9, save_model=False):
    """
    Description
    -----------
    This function grouped together the different elements needed to execute the model
    training and trains the model. After the training plots the loss and the
    accuracy of both the train and the test. To finish returns the trained
    model.
    
    Parameters
    ----------
    model_to_train : class
        The neural network model to be trained.
    model_name: str
        The name of the model to train
    dataloaders : dict of DataLoader
        A dictionary containing the data loaders for the training and validation sets.
    epochs : int, optional
        Define the number of epochs to train the model. The default is 25.
    learning_rate : float, optional
        Defines the learning rate of the optimizer. The default is 0.001.
    momentum : float, optional
        Defines the momentum of the optimizer. The default is 0.9.
    save_model: bool, optional
        If true saves the best weights of the trained model. The default is False.
        
    Returns
    -------
    trained_model: class
        The trained neural network

    """

    # initializing optimizer and criterion
    optimizer = optim.SGD(model_to_train.parameters(),
                          lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # loading model to device and training
    model_to_train = model_to_train.to(device)
    trained_model, hist, losses = train(model_to_train, dataloaders,
                                        criterion, optimizer, num_epochs=epochs)
    model_metrics_plot(hist, losses)

    if(save_model):
        torch.save(model.state_dict(),("./models/"+model_name+".pth"))
        print("Model saved.")
    
    return trained_model


if __name__ == "__main__":
    batch_size=12
    num_workers=4
    data_path="./Dog-Breed-classif"
    dataloaders = dogs_dataset_dataloders(data_transforms_complete, data_path,
                                          batch_size, num_workers)

    model, params, model_name = initialize_model_resnext101_32x8d(120)
    trained_model = training_pipeline(model, model_name, dataloaders, epochs=20,
                                      save_model=True)

