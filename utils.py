import torch
import matplotlib
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim


matplotlib.style.use('ggplot')      #ggplot: a popular plotting style from R


def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    to save the trained model to disk
    """
    torch.save({
        'epoch' : epochs,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'loss' : criterion,
    }, f"../outputs/model_pretrained_{pretrained}.pth")


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    """
    to save the loss and accuracy plots to disc
    """

    #accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color = 'green', linestyle = '-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color = 'blue', linestyle = '-',
        label = 'validation accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"../outputs/accuracy_pretrained_{pretrained}.png")

    #Loss plots
    plt.figure(figsize = (10, 7))
    plt.plot(
        train_loss, color = 'orange', linestyle = '-',
        label = 'train loss'
    )
    plt.plot(
        valid_loss, color = 'red', linestyle = '-',
        label = 'validation loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../outputs/loss_pretrained_{pretrained}.png")


"""if __name__ == "__main__":
    
    # Ensure output directory exists
    os.makedirs("../outputs", exist_ok=True)

    # Dummy model
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Test save_model
    save_model(
        epochs=5,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        pretrained=True  # or False, to test both variants
    )
    print("Model saved successfully.")

    # Dummy accuracy/loss data
    train_acc = [60, 65, 70, 75, 80]
    valid_acc = [55, 60, 68, 72, 76]
    train_loss = [1.0, 0.8, 0.6, 0.5, 0.4]
    valid_loss = [1.1, 0.9, 0.7, 0.6, 0.5]

    # Test save_plots
    save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained=True)
    print("Plots saved successfully.")
"""