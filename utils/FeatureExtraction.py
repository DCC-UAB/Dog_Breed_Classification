import matplotlib.pyplot as plt
import torch.nn as nn


def feature_extraction(model, img):
    """
    Description
    -----------
    Performs feature extraction from a convolutional neural network using the PyTorch 
    library and visualizes the outputs of each convolutional layer.
    
    Parameters
    ----------
    model: class
       contain the convolutional neural network
    img: PIL image
        input image to do the feature extraction
    
    Returns
    -------
    """
    # Initialize variables
    no_of_layers=0
    conv_layers=[]
 
    # Get the children of the model
    model_children=list(model.children())
    
    # Iterate over each child of the model
    for child in model_children:
        # If the child is a Conv2d layer, increment the layer counter and add it to the list
        if type(child)==nn.Conv2d:
            no_of_layers+=1
            conv_layers.append(child)
        
        # If the child is a Sequential layer, iterate over its children and check for Conv2d layers
        elif type(child)==nn.Sequential:
            for layer in child.children():
                if type(layer)==nn.Conv2d:
                    no_of_layers+=1
                    conv_layers.append(layer)
    # Print the total number of convolutional layers found
    print(no_of_layers)
    
    # Apply the first convolutional layer to the input image and store the result
    results = [conv_layers[0](img)]
    
    # Iterate over the remaining convolutional layers
    for i in range(1, len(conv_layers)):
        # Apply each layer to the output of the previous layer and store the result
        results.append(conv_layers[i](results[-1]))
    
    outputs = results
    
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(50, 10))
        # Get the visualization of the current layer's output
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print("Layer ",num_layer+1)
       
    # Show only the first 16 filters
        for i, filter in enumerate(layer_viz):
            if i == 16: 
                break
            plt.subplot(2, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        # Show the current figure
        plt.show()
        plt.close()
