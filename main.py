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
from models.models import initialize_model_regnet_x_16
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def training_pipeline(model_to_train, epochs=10,  batch_size=12, num_workers=4,
                      learning_rate=0.001, data_path="./Dog-Breed-classif"):

    # creating dataloaders
    dataloaders = dogs_dataset_dataloders(data_transforms_complete, batch_size, num_workers, data_path)

    # initializing optimizer and criterion
    optimizer = optim.SGD(model_to_train.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # loading model to device and training
    model_to_train = model_to_train.to(device)
    trained_model, hist, losses = train(model_to_train, dataloaders, criterion, optimizer, num_epochs=epochs)
    model_metrics_plot(hist, losses)
    return trained_model


if __name__ == "__main__":
    print(os.getcwd())
    model, params = initialize_model_regnet_x_16(120)
    training_pipeline(model,epochs=15)

