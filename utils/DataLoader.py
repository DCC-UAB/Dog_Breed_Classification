# Imports
import random
import torch
import torchvision
import pandas as pd

"""
This Python code defines a custom Dataset class in PyTorch and a function to load, process, and split a dataset of dog images into 
training, validation, and test sets. It also creates PyTorch dataloaders for each of these sets, which are used to efficiently 
load the data during training of a neural network. The function uses CUDA for better performance if available.
"""

# Definition of a custom dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset  # Subset of data
        self.transform = transform  # Data augmentation/transformations

    def __getitem__(self, index):
        x, y = self.subset[index]  # Get an item by index
        if self.transform:  # If any transformations were passed
            x = self.transform(x)  # Apply transformations
        return x, y

    def __len__(self):
        return len(self.subset)  # Return the length of subset

# Function to load data and return dataloaders for each data subset
def dogs_dataset_dataloders(transformer, batch_size, num_workers, dataset_path, shuffle=True):

    # Use CUDA if available for performance enhancement
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    # Load data
    dataset = torchvision.datasets.ImageFolder(dataset_path+'/train')
    labels = pd.read_csv(dataset_path+'/labels.csv')

    # Split data into training, validation and testing subsets
    dataset_len = labels.shape[0]
    indexes = list(range(dataset_len))
    random.shuffle(indexes)
    train_indexes = indexes[0:int(dataset_len * 0.6)]
    validation_indexes = indexes[int(dataset_len * 0.6) + 1: int(dataset_len * 0.9)]
    test_indexes = indexes[int(dataset_len * 0.9) + 1: (dataset_len-1)]
    train_subset = torch.utils.data.Subset(dataset, train_indexes)
    train_dataset = MyDataset(train_subset, transform=transformer["train"])
    validation_subset = torch.utils.data.Subset(dataset, validation_indexes)
    validation_dataset = MyDataset(validation_subset, transform=transformer["val"])
    test_subset = torch.utils.data.Subset(dataset, test_indexes)
    test_dataset = MyDataset(test_subset, transform=transformer["test"])

    # Create dataloaders for each subset
    dataloaders_dict = {"train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                             shuffle=shuffle, num_workers=num_workers),
                        "val": torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                           shuffle=shuffle, num_workers=num_workers),
                        "test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                            shuffle=shuffle, num_workers=num_workers)
                        }

    return dataloaders_dict
