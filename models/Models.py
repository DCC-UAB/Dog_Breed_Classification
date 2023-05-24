#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[12]:


import torch.nn as nn
import torchvision.models as models
from keras.models import Model
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Sequential
from keras.applications import xception

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

# In[2]:


# Conventional and convolutional neural network

class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        input_size = 224
        return out, input_size


# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN

        # Convolution layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Max pooling layer (divides image size by 2)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 500)
        self.fc2 = nn.Linear(500, 120)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        ## Define forward behavior

        # Sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten image input
        x = x.view(-1, 128 * 28 * 28)
        # Dropout layer
        x = self.dropout(x)
        # 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # Dropout layer
        x = self.dropout(x)
        # 2nd hidden layer
        x = self.fc2(x)
        input_size = 224
        return x, input_size


class XV3V2N(nn.Module):

    def __init__(self):
        super(XV3V2N, self).__init__()
        self.base_model_1 = models.alexnet(weights='DEFAULT')
        self.base_model_2 = models.inception_v3(weights='DEFAULT')
        self.base_model_3 = models.resnext50_32x4d(weights='DEFAULT')
        self.base_model_5 = models.mnasnet1_0(weights='DEFAULT')

        self.base_model_1.fc = nn.Identity()
        self.base_model_2.fc = nn.Identity()
        self.base_model_3.fc = nn.Identity()
        self.base_model_5.fc = nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(2688, 120)


    def forward(self, inputs):
        ## <-----  Xception   -----> ##
        x1 = self.base_model_1(inputs)
        ## <-----  InceptionV3   -----> ##
        x2 = self.base_model_2(inputs)
        ## <-----  InceptionResNetV2   -----> ##
        x3 = self.base_model_3(inputs)
        ## <-----  NASNetLarge   -----> ##
        x5 = self.base_model_5(inputs)

        x1 = self.pool(x1).view(x1.size(0), -1)
        x2 = self.pool(x2).view(x2.size(0), -1)
        x3 = self.pool(x3).view(x3.size(0), -1)
        x5 = self.pool(x5).view(x5.size(0), -1)

        x = torch.cat([x1, x2, x3, x5], dim=1)
        x = self.dropout(x)
        outputs = self.fc(x)

        return outputs


# https://www.kaggle.com/code/khanrahim/dog-breed
#
# val_accuracy = 90%

# In[8]:


class ConcatIXIRN(nn.Module):

    def __init__(self):
        super(ConcatIXIRN, self).__init__()

    def get_features(self, model_name, data_preprocessor, input_size, data):
        # Function to extract features from images
        # Prepare pipeline.
        input_layer = Input(input_size)
        preprocessor = Lambda(data_preprocessor)(input_layer)
        base_model = model_name(weights='imagenet', include_top=False,
                                input_shape=input_size)(preprocessor)
        avg = GlobalAveragePooling2D()(base_model)
        feature_extractor = Model(inputs=input_layer, outputs=avg)
        # Extract feature.
        feature_maps = feature_extractor.predict(data, batch_size=32, verbose=1)
        print('Feature maps shape: ', feature_maps.shape)
        return feature_maps

    def forward(self, img_size):
        # Extracting features using InceptionV3
        inception_preprocessor = preprocess_input
        inception_features = self.get_features(InceptionV3,
                                               inception_preprocessor,
                                               img_size, X)

        # Extracting features using Xception
        xception_preprocessor = preprocess_input
        xception_features = self.get_features(Xception,
                                              xception_preprocessor,
                                              img_size, X)

        # Extracting features using InceptionResnetV2
        inc_resnet_preprocessor = preprocess_input
        inc_resnet_features = self.get_features(InceptionResNetV2,
                                                inc_resnet_preprocessor,
                                                img_size, X)

        # concatinating features
        final_features = np.concatenate([inception_features,
                                         xception_features,
                                         inc_resnet_features, ], axis=-1)
        print('Final feature maps shape', final_features.shape)

        # Building Model
        model = Sequential()
        model.add(InputLayer(final_features.shape[1:]))
        model.add(Dropout(0.7))
        model.add(Dense(120, activation='softmax'))
        input_size = 224
        return model, input_size


# Compiling Model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# # LogReg on Xception bottleneck features
#
# https://www.kaggle.com/code/gaborfodor/dog-breed-pretrained-keras-models-lb-0-3#LogReg-on-Xception-bottleneck-features
#
# Val_accuracy = 98 %

#
