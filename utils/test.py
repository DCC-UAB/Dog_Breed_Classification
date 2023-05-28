import pandas as pd
from tqdm import tqdm
import re
import torch.nn as nn
import torch

# Falta importar els labels
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, dataloader):
    model.eval()
    predictions = pd.DataFrame(columns=["label", "prediction"])
    with torch.no_grad():
        for data, labels in tqdm(dataloader):
            data = data.to(device)
            labels = [int(i) for i in list(labels.cpu())]
            output = model(data)
            _, result = torch.max(output, 1)
            result = result.cpu()
            df = {"label": labels, "prediction": result}
            temp_df = pd.DataFrame(df)
            predictions = pd.concat([predictions, temp_df], ignore_index=True)

    predictions["correct"] = (predictions["label"] == predictions["prediction"])
    accuracy = sum(predictions["correct"])/len(predictions)
    return predictions, accuracy


def test_on_fold(model, test_loader, model_name="", optimizer_name="", load_weights=False):
    if load_weights:
        path = "./models/"+model_name+"_"+optimizer_name+".pth"
        model.load_state_dict(torch.load(path, device))

    predictions, accuracy = test(model, test_loader)
    

    return predictions, accuracy

