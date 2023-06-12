import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image

def heatmap_32x8(path):
    """
    Description
    -----------
    Generates a class activation map (CAM) using the GradCAM algorithm for a given input image.
    The input image is preprocessed, and GradCAM is applied to obtain the class activation map. 
    The CAM is then overlaid on the input image. 
    
    Parameters
    ----------
    path: str
       contain the path of the image 
    
    Returns
    -------
    The function returns the resulting image as a PIL image object.
    """
    
    # Load the pre-trained ResNeXt-101 32x8d model
    model = models.resnext101_32x8d(pretrained=True)
    model.eval()
    img = np.array(Image.open(path))
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Set the target class for visualization
    targets = [ClassifierOutputTarget(295)]
    
    # Specify the target layer for GradCAM visualization
    target_layers = [model.layer4]
    
    # Create GradCAM object
    with GradCAM(model=model, target_layers=target_layers) as cam:
        # Generate the gradient-based class activation map
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        
        # Overlay the class activation map on the input image
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

    # Convert the grayscale CAM to color format
    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    
    # Create a horizontal stack of original image, CAM, and CAM overlaid on the input image
    images = np.hstack((np.uint8(255*img), cam , cam_image))
   
    # Convert the NumPy array to a PIL image and return it
    Image.fromarray(images)
    return(images)
