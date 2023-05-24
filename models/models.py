#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision


# parameter extraction
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model_resnet_18(num_classes, pre_trained=True,
                               feat_extraction=True):
    # Resnet18
    model = torchvision.models.resnet18(pretrained=pre_trained)
    if(pre_trained):
        model_name="ResNet_18_PreTrained"
    else:
        model_name="ResNet_18"
    set_parameter_requires_grad(model, feat_extraction)
    model.fc = nn.Linear(512, num_classes)
    input_size = 224

    return model, input_size, model_name

def initialize_model_resnext101_64x4d(num_classes, pre_trained=True,
                                      feat_extraction=True):
    # ResNext101_32x8d
    if(pre_trained):
        weights = torchvision.models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        model = torchvision.models.resnext101_64x4d(weights=weights)
        model_name="RexNext101_64x4d_PreTrained"
    else:
        model = torchvision.models.resnext101_64x4d()
        model_name="ResNext101_64x4d"
        
    set_parameter_requires_grad(model, feat_extraction)
    model.fc = nn.Linear(2048, num_classes)
    input_size = 224

    return model, input_size, model_name


def initialize_model_resnext101_32x8d(num_classes, pre_trained=True,
                                      feat_extraction=True):
    # ResNext101_32x8d
    model = torchvision.models.resnext101_32x8d(pretrained=pre_trained)
    if(pre_trained):
        model_name="ResNext101_32x8d_PreTrained"
    else:
        model_name="ResNext101_32x8d"
    set_parameter_requires_grad(model, feat_extraction)
    model.fc = nn.Linear(2048, num_classes)
    input_size = 224

    return model, input_size, model_name

def initialize_model_resnext50_32x4d(num_classes, pre_trained=True,
                                      feat_extraction=True):
    # ResNext101_32x8d
    model = torchvision.models.resnext50_32x4d(pretrained=pre_trained)
    if(pre_trained):
        model_name="ResNext50_32x4d_PreTrained"
    else:
        model_name="ResNext50_32x4d"
    set_parameter_requires_grad(model, feat_extraction)
    model.fc = nn.Linear(2048, num_classes)
    input_size = 224

    return model, input_size, model_name


