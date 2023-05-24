# Imports
from .train import *
from .test import *
from models.models import *
import sys

"""
The code provided aims to make the execution of the models user-friendly. Making the training and testing easily callable
"""

# Main function to drive the menu
def menu (dataloaders):
    # Print the menu options
    print("#############################\n"
          "\tDog Breed Classifier\n"
          "#############################")
    print("Please input a mode:\n- Only train: 1\n- Only test: 2\n- Train and Test: 3\n")
    mode = int(input())  # Take user input
    model, params, model_name = select_model()  # Select model
    optimizer = select_optimizer(model.parameters())  # Select optimizer
    if mode in [1, 3]:  # If training is involved, ask for number of epochs
        print("Input number of epochs:")
        epochs = int(input())

    model = model.to(device)# Move model to GPU if available
    if mode == 1: # Only train
        print("Do you want to save the model weights after training? [y/n]")
        save_model = input()
        trained_model = training_pipeline(model, model_name, dataloaders, optimizer,
                                          epochs, save_model == "y")
    elif mode == 2: # Only test
        predictions, accuracy = test_on_fold(model, dataloaders['test'], model_name, "SGD")
        print("Test accuracy:", accuracy)

    elif mode == 3: # Train and test
        trained_model = training_pipeline(model, model_name, dataloaders, optimizer,
                                          epochs, save_model=True)
        predictions, accuracy = test_on_fold(trained_model, dataloaders['test'], load_weights=False)

        print("Test accuracy:", accuracy)

    else: # Invalid mode input
        print("Invalid mode, exiting...")
        sys.exit()

# Function to select optimizer and its hyperparameters
def select_optimizer(parameters):
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

    else: # Invalid optimizer input
        print("Invalid optimizer, exiting...")
        sys.exit()

    return optimizer

# Function to select the model, with user deciding whether it's pretrained and used for feature extraction
def select_model():
    model = None
    feat_extraction = "n"
    num_classes = 120
    # Models defined in Models.py
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
