#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision


# parameter extraction
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model_regnet_x_16(num_classes):
    # Resnet18
    model = torchvision.models.regnet_x_16gf(pretrained=True)
    set_parameter_requires_grad(model, True)
    model.fc = nn.Linear(2048, num_classes)
    input_size = 224

    return model, input_size

def initialize_model_regnet_x_16(num_classes):
    # Resnet18
    model = torchvision.models.regnet_x_16gf(pretrained=True)
    set_parameter_requires_grad(model, True)
    model.fc = nn.Linear(2048, num_classes)
    input_size = 224

    return model, input_size

def initialize_model_regnet_x_16(num_classes):
    # Resnet18
    model = torchvision.models.regnet_x_16gf(pretrained=True)
    set_parameter_requires_grad(model, True)
    model.fc = nn.Linear(2048, num_classes)
    input_size = 224

    return model, input_size


def initialize_model_resnext101_32x8d(num_classes):
    # ResNext101_32x8d
    model = torchvision.models.resnext101_32x8d(pretrained=True)
    model_name="ResNext101_32x8d"
    set_parameter_requires_grad(model, True)
    model.fc = nn.Linear(2048, num_classes)
    input_size = 224

    return model, input_size, model_name



