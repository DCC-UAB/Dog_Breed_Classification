# Imports
import matplotlib.pyplot as plt
"""
This code is used to make a visual analysis of the models by ploting the loss and acurracy of the training and validation sets.
"""
def model_metrics_plot(acc_history, losses):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    plt.suptitle("Metrics evolution during model training")
    axs = axs.flatten()

    axs[0].plot(losses["train"], label="Training loss")
    axs[0].plot(losses["val"], label="Validation loss")
    axs[0].legend()

    axs[1].plot(acc_history["train"], label="Training accuracy")
    axs[1].plot(acc_history["val"], label="Validation accuracy")
    axs[1].legend()

    plt.show()

    return fig