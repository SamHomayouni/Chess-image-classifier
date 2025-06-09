import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots


#the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e',
    '--epochs',
    type= int,
    default=20,
    help= 'Number of epochs to train each network for'
    )
parser.add_argument(
    '-pt',
    '--pretrained',
    action= 'store_true',
    help= 'Whether to use pretrained weights or not'
    )
parser.add_argument(
    '-lr',
    '--learning-rate',
    type= float,
    dest= 'learning_rate',
    default= 0.0001,
    help= 'Learning rate for training the model'
    )
args = vars(parser.parse_args())    #converts the parsed command-line args from a namespace object into a dict.


#training function
def train(model, trainloader, optimizer, criterion):
    model.train()   #put the model in training mode
    print('Training')
    #initializing accumulators
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        #moving input tensors in gpu
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()   #clearing old grads
        outputs = model(image)  #forward pass
        loss = criterion(outputs, labels)   #calculate the loss
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)   #calculate the accuracy
        #adding num of correct preds in a batch
        train_running_correct += (preds == labels).sum().item()
        loss.backward()     #backpropagation
        optimizer.step()    #update the weights

    #loss and acc for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc



#validation function
def validate(model, testloader, criterion):
    model.eval()
    print('Validating')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad(): #no need for grad tracking in eval mode
        for i, data in tqdm(enumerate(testloader), total= len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    #load the training and validating datasets
    dataset_train, dataset_valid, dataset_classes = get_datasets(args['pretrained'])
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    #load the training and validation data loaders
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
    #learning params
    lr = args['learning_rate']
    epochs = args['epochs']
    #check if a gpu is available, if yes then use cuda if no use cpu
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    
    model = build_model(
        pretrained= args['pretrained'],
        fine_tune= True,
        num_classes= len(dataset_classes)
    ).to(device)

    #total params and trainable params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters")
    total_trainable_params = sum(
        p.numel() for p in model.parameters if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters")

    #optimizer
    optimizer = optim.Adam(model.parameters, lr= lr)
    #loss function
    criterion = nn.CrossEntropyLoss()

    #keeping track of losses and accs in lists
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    #start the training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model= model,
            trainloader= train_loader,
            optimizer= optimizer,
            criterion= criterion)
        
        valid_epoch_loss, valid_epoch_acc = train(
            model= model,
            trainloader= valid_loader,
            criterion= criterion)
        
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training accuracy: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation accuracy: {valid_epoch_acc:.3f}")
        print('-'*50)
        time.sleep(5)

    #save the trained model weights
    save_model(epochs= epochs,
               model= model,
               optimizer= optimizer,
               criterion= criterion,
               pretrained= args['pretrained'])
    #save the loss and accuracy plots
    save_plots(train_acc= train_acc,
               valid_acc= valid_acc,
               train_loss= train_loss,
               valid_loss= valid_loss,
               pretrained= args['pretrained'])
    print('TRAINING COMPLETE')
