# Imports
import random
import torch
import torchvision
import pandas as pd

class MyDataset(torch.utils.data.Dataset):
    
    # Constructor: takes subset (data) and transform (optional data transform function).
    def __init__(self, subset, transform=None):
        self.subset = subset  
        self.transform = transform  

    # Returns the item at the given index, applies transform if it exists.
    def __getitem__(self, index):
        x, y = self.subset[index]  
        if self.transform:  
            x = self.transform(x)
        return x, y  

    # Returns the size of the dataset.
    def __len__(self):
        return len(self.subset)  


def dogs_dataset_dataloders(transformer, dataset_path, batch_size=12,
                            num_workers=4, shuffle=True):
    """
    Description
    -----------
    Generates a dictionary with three dataloaders: train, val and test.
    Parameters
    ----------
    transformer : dict of torchvision.transforms.compose
        Contains the transformations to apply on the data loaded by the
        dataloaders
    dataset_path : str
        Contains the relative or absolute path to the folder with the images
        and the labels of the kaggle.
    batch_size : int, optional
        How many samples per batch to load. The default is 12.
    num_workers : int, optional
        how many subprocesses to use for data loading, 0 means that the data
        will be loaded in the main process. The default is 4.
    shuffle : bool, optional
        If true the dataloaders shuffle the data they have. The default is true.

    Returns
    -------
    dataloaders_dict: dict of dataloader
        Contains the tree dataloaders for each step of the training, validation
        and test of a neural network.

    """

    # activate cuda for performance enhacement
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    # loading data
    dataset = torchvision.datasets.ImageFolder(dataset_path + '/train')
    labels = pd.read_csv(dataset_path + '/labels.csv')

    # splitting data
    dataset_len = labels.shape[0]
    indexes = list(range(dataset_len))
    random.shuffle(indexes)
    train_indexes = indexes[0:int(dataset_len * 0.6)]
    validation_indexes = indexes[int(dataset_len * 0.6) + 1: int(dataset_len * 0.9)]
    test_indexes = indexes[int(dataset_len * 0.9) + 1: (dataset_len - 1)]
    train_subset = torch.utils.data.Subset(dataset, train_indexes)
    train_dataset = MyDataset(train_subset, transform=transformer["train"])
    validation_subset = torch.utils.data.Subset(dataset, validation_indexes)
    validation_dataset = MyDataset(validation_subset, transform=transformer["val"])
    test_subset = torch.utils.data.Subset(dataset, test_indexes)
    test_dataset = MyDataset(test_subset, transform=transformer["test"])


    #creating dataloaders
    dataloaders_dict = {"train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                             shuffle=shuffle, num_workers=num_workers),
                        "val": torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                           shuffle=shuffle, num_workers=num_workers),
                        "test": torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                            shuffle=shuffle, num_workers=num_workers)
                        }

    return dataloaders_dict
