import random
import torch
import torchvision
import pandas as pd

def DogsDatasetDataloders(batch_size, num_workers, shuffle=True):
    #activate cuda for performance enhacement
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    #loading data
    dataset = torchvision.datasets.ImageFolder('../Dog-Breed-classif/train', torchvision.transforms.ToTensor())
    labels = pd.read_csv("../Dog-Breed-classif/labels.csv")

    #splitting data
    dataset_len = labels.shape[0]
    indexes = list(range(dataset_len))
    random.shuffle(indexes)
    train_indexes = indexes[0:int(dataset_len*0.7)]
    validation_indexes = indexes[int(dataset_len*0.7)+1:dataset_len-1]
    train_subset = torch.utils.data.Subset(dataset, train_indexes)
    validation_subset = torch.utils.data.Subset(dataset, validation_indexes)

    dataloaders_dict={"train": torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
                      "val": torch.utils.data.DataLoader(validation_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)}

    return dataloaders_dict