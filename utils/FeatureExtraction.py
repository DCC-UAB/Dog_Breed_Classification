import matplotlib.pyplot as plt
import torch.nn as nn


def feature_extraction(model, img):
    no_of_layers=0
    conv_layers=[]
 
    model_children=list(model.children())
 
    for child in model_children:
        if type(child)==nn.Conv2d:
            no_of_layers+=1
            conv_layers.append(child)
        elif type(child)==nn.Sequential:
            for layer in child.children():
                if type(layer)==nn.Conv2d:
                    no_of_layers+=1
                    conv_layers.append(layer)
    print(no_of_layers)
    results = [conv_layers[0](img)]
    
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))
    outputs = results
    
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(50, 10))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print("Layer ",num_layer+1)
        for i, filter in enumerate(layer_viz):
            if i == 16: 
                break
            plt.subplot(2, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        plt.show()
        plt.close()