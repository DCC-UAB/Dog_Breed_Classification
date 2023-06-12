from .train import *
from .test import *
from models.models import *
import sys


def menu(dataloaders):
    """
    Description
    -----------
    This function is used as a way facilitate the navigation through the different phases of training and testing
    with some implemented models and optimizers in our project.
    
    Parameters
    ----------
    dataloaders: dictionary of torch.utils.data.DataLoader
        Contains the dataloaders for the three phases of the menu. Train, validation and test.
    Returns
    -------
    trained_model: class
        Contains the trained model
    predictions, accuracy: Pandas.DataFrame, float
        Contains the predictions done by our model during the testing phase and the accuracy of the test.
    """
    
    print("#############################\n"
          "\tDog Breed Classifier\n"
          "#############################")
    print("Please input a mode:\n- Only train: 1\n- Only test: 2\n- Train and Test: 3\n")
    mode = int(input())
    if mode not in [1, 2, 3]:
        print("Invalid mode, exiting...")
        sys.exit()

    model, params, model_name = select_model()
    optimizer = select_optimizer(model.parameters())
    optimizer_name = type(optimizer).__name__
    if mode in [1, 3]:
        print("Input number of epochs:")
        epochs = int(input())
    model = model.to(device)

    if mode == 1:
        print("Will you want to save the model weights after training? [y/n]")
        save_model = input()
        print("Will you want to save the metrics plot? [y/n]")
        save_plot = input()
        trained_model = training_pipeline(model, model_name, dataloaders, optimizer,
                                          epochs, save_model == "y", save_plot == "y")

        return trained_model

    elif mode == 2:
        predictions, accuracy = test_on_fold(model, dataloaders['test'], model_name=model_name,
                                             optimizer_name=optimizer_name, load_weights=True)
        print("Test accuracy:", accuracy)

        return predictions, accuracy

    elif mode == 3:
        print("Will you want to save the model weights after training? [y/n]")
        save_model = input()
        print("Will you want to save the metrics plot? [y/n]")
        save_plot = input()
        trained_model = training_pipeline(model, model_name, dataloaders, optimizer,
                                          epochs, save_model == "y", save_plot == "y")
        predictions, accuracy = test_on_fold(trained_model, dataloaders['test'], load_weights=False)

        print("Test accuracy:", accuracy)

        return predictions, accuracy


def select_optimizer(parameters):
    """
    Description
    -----------
    This function is used as a way facilitate the initialization of the implemented optimizer that is going to be
    used during the training.
    
    Parameters
    ----------
    parameters: iterator [torch.nn.parameter.Parameter]
        Iterator that contains the parameters of the model for the optimizer initialization.
    Returns
    -------
    optimizer: torch.optim
        The optimizer initializated with the desired parameters
    """
        
    optimizer = None
    optimizers = ["SGD", "Adam", "RMSprop"]

    print("Select one of the implemented optimizers:")
    for i, opt_name in enumerate(optimizers):
        print("- ", opt_name, ": ", i+1)
    opt_num = int(input())
    print("Input a learning rate: ")
    learning_rate = float(input())

    if opt_num == 1:
        print("Input a momentum: ")
        momentum = float(input())
        optimizer = optim.SGD(parameters, lr=learning_rate,
                              momentum=momentum)
    elif opt_num == 2:
        optimizer = optim.Adam(parameters, lr=learning_rate)

    elif opt_num == 3:
        print("Input a momentum: ")
        momentum = float(input())
        print("Input an alpha: ")
        alpha = float(input())
        optimizer = optim.RMSprop(parameters, lr=learning_rate,
                                  momentum=momentum, alpha=alpha)

    else:
        print("Invalid optimizer, exiting...")
        sys.exit()

    return optimizer


def select_model():
    """
    Description
    -----------
    This function is used as a way facilitate the initialization of the implemented model that is going to be
    used during the training or testing.
    
    Parameters
    ----------
    None.
    
    Returns
    -------
    model: class
        The model initializated with the desired parameters
    """
        
    model = None
    feat_extraction = "n"
    num_classes = 120
    models = ["ResNext101_32x8d", "ResNext101_64x4d", "ResNext50_32x4d",
              "Resnet18"]
    print("Select one of the implemented models:")
    for i, model_name in enumerate(models):
        print("- ", model_name, ": ", i+1)
    model_num = int(input())

    print("Do you want it to be pretrained? [y/n]")
    pretrained = input()
    if pretrained == "y":
        print("Do you want to do feature extraction? [y/n]")
        feat_extraction = input()

    if model_num == 1:
        model = initialize_model_resnext101_32x8d(num_classes, pretrained == "y",
                                                  feat_extraction == "y")
    elif model_num == 2:
        model = initialize_model_resnext101_64x4d(num_classes, pretrained == "y",
                                                  feat_extraction == "y")
    elif model_num == 3:
        model = initialize_model_resnext50_32x4d(num_classes, pretrained == "y",
                                                 feat_extraction == "y")
    elif model_num == 4:
        model = initialize_model_resnet_18(num_classes, pretrained == "y",
                                           feat_extraction == "y")
    else:
        print("Invalid model, exiting menu...")
        sys.exit()

    return model
