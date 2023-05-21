import random
import torch
import torchvision
import pandas as pd

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)


def dogs_dataset_dataloders(transformer, batch_size, num_workers, dataset_path, shuffle=True):

    #activate cuda for performance enhacement
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    #loading data
    dataset = torchvision.datasets.ImageFolder(dataset_path+'/train')
    labels = pd.read_csv(dataset_path+'/labels.csv')

    # splitting data
    dataset_len = labels.shape[0]
    indexes = list(range(dataset_len))
    random.shuffle(indexes)
    train_indexes = indexes[0:int(dataset_len * 0.7)]
    validation_indexes = indexes[int(dataset_len * 0.7) + 1:dataset_len - 1]
    train_subset = torch.utils.data.Subset(dataset, train_indexes)
    train_dataset = MyDataset(train_subset, transform=transformer["train"])
    validation_subset = torch.utils.data.Subset(dataset, validation_indexes)
    validation_dataset = MyDataset(validation_subset, transform=transformer["val"])

    #creating dataloaders
    dataloaders_dict = {"train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
                        "val": torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)}

    return dataloaders_dict