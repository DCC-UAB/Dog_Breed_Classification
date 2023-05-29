import pandas as pd
from tqdm import tqdm
import torch

# Falta importar els labels
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, dataloader):
    """
    Description
    -----------
    Does the testing phase of a model though the test dataloader to give more information about the model
    that we have trained and then returns the information we got of this test.
    
    Parameters
    ----------
    model: class
        Contains the trained model we want to test.
    dataloader: dictionary of torch.utils.data.DataLoader
        Contains the dataloader of the testing phase.
        
    Returns
    -------
    predictions: Pandas.DataFrame
        Contains the original labels, the predicted ones and if the prediction was right.
    accuracy: float
        The computed accuracy of our test
    """
    
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
    """
    Description
    -----------
    Joins the weight loading with the test function
    
    Parameters
    ----------
    model: class
        Contains the trained model we want to test.
    test_dataloader: dictionary of torch.utils.data.DataLoader
        Contains the dataloaders of the testing phase.
    model_name: str
        The model name, used to open the weights file. By default "".
    optimizer_name: str
        The optimizer name, used to open the weights file. By default "".
    load_weights: bool
        If true loads the weights of a previously trained model. By default False.
      
    Returns
    -------
    predictions: Pandas.DataFrame
        Contains the original labels, the predicted ones and if the prediction was right.
    accuracy: float
        The computed accuracy of our test
    """
        
    if load_weights:
        path = "./models/"+model_name+"_"+optimizer_name+".pth"
        model.load_state_dict(torch.load(path, device))

    predictions, accuracy = test(model, test_loader)

    return predictions, accuracy

