import torch
import numpy as np
import time
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, dataloaders, criterion, optimizer, num_epochs=25):
    """
    Description
    -----------
    This function is used to train a given neural network model. During training, 
    it iterates over the specified number of epochs and performs both training 
    and validation steps. It tracks the training and validation losses, as well 
    as the accuracy for each epoch. The best model weights based on validation 
    accuracy are saved and used for evaluation.
    
    Parameters
    ----------
    model : class
        The neural network model to be trained.
    dataloaders : TYPE
        A dictionary containing the data loaders for the training and validation sets.
    criterion : torch.nn
        The loss function used to calculate the loss.
    optimizer : torch.optim
        The optimization algorithm used to update the model's weights.
    num_epochs : int, optional
        The number of epochs to train the model. The default is 25.

    Returns
    -------
    model : class
        The trained model.
    acc_history : dictionary
        The dictionary of the accuracy history for train and val.
    losses : dictionary
        The dictionary of the losses history for train and val.

    """
    #stores the current time to calculate the total training time later
    since = time.time()

    #Dictionary that will store the training and validation accuracies for each epoch
    acc_history = {"train": [], "val": []}
    
    #Dictionary that will store the training and validation losses for each epoch.
    losses = {"train": [], "val": []}

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            #Variables that keep track of the cumulative loss and the number of correct predictions in the current epoch
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data using the data loader corresponding to the current phase
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.item())

                    #The predictions are obtained by taking the maximum value along the appropriate dimension of the outputs.
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            #Average loss per data point in the current phase
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            #number of correct predictions divided by the dataset size in the current phase
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            #Print to see the results
            print('Phase: {} \tTraining Loss: {:.4f} \t Accuracy: {:.4f}'.format(
            phase, 
            epoch_loss,
            epoch_acc
            ))
            

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
           
            
            acc_history[phase].append(epoch_acc.item())

        print()

    #Calculate and print the total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_history, losses
