from torchvision import transforms

#Global variables

#Image normalization variable
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

input_size = 224 #Size of the input images
    
#Transformation 1 (Basic)
data_transforms_basic = {
    'train': transforms.Compose([
        transforms.Resize(input_size), #Resize the image to the specified input size
        transforms.CenterCrop(input_size), #Center crop the image
        transforms.ToTensor(), #Convert to tensor Pytorch tensor
        normalize #Apply the normalize
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


#Transformation 2 (Grey Scale)
data_transforms_gray_scale = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.Grayscale(3), #Convert the image to grey scale output channels 3, RGB but in gray scale
        transforms.ToTensor() # 

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

#Transformation 3 (Complete one)
data_transforms_complete = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomVerticalFlip(p=0.5), #Random flip the image vertically with a probablity of 0.5
        transforms.RandomHorizontalFlip(p=0.5), #Random flip the image horizontally with a probablity of 0.5
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Apply Gaussian blur to the image
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

#Transformation 4 (jitter)
data_transforms_jitter = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ColorJitter(brightness=.5, hue=.3), # Apply color jitter to the image
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
