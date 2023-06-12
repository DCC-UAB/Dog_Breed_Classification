# Imports
from torchvision import transforms

# Global variables
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

input_size = 224

# Transformation 1 (Basic)
data_transforms_basic = {
    'train': transforms.Compose([
        transforms.Resize(input_size),  # Establim la mida a 224
        transforms.CenterCrop(input_size),  # Centrem el tallat de les imatges
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
}

# Transformation 2 (Grey Scale)
data_transforms_gray_scale = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.Grayscale(3),  # Output channels 3, RGB but in gray scale
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
}

# Transformation 3 (Complete)
data_transforms_complete = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomVerticalFlip(p=0.5),  # Girem verticalment
        transforms.RandomHorizontalFlip(p=0.5),  # Girem horitzontalment
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apliquem un blur
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
}

# Transformation 4 (jitter)
data_transforms_jitter = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ColorJitter(brightness=.5, hue=.3),  # Jitter color
        transforms.ColorJitter(brightness=.5, hue=.3),  # Jitter color
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ]),
}