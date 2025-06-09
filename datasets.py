import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

ROOT_DIR = '/Users/shomayouni/Desktop/EfficientNet/input/Chess'
VALID_SPLIT = 0.1
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4     #number of parallel processes for data preparation


#training transforms:
def get_train_transform(IMAGE_SIZE, pretrained):
    train_trnsform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomAdjustSharpness(sharpness_factor= 2, p = 0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_trnsform

#validation transform:
def get_valid_transform(IMAGE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform

#image normalization transforms:
def normalize_transform(pretrained):
    if pretrained:  #for pretrained weights which I will use
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )

    else:   #for training from scratch that I am not using here but in just case
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std = [0.5, 0.5, 0.5]
        )
    return normalize

def get_datasets(pretrained):
    """
    to prepare the dataset.

    :param pretrained: Boolean.

    Returns the training and validation dayasets along with the class names
    """
    dataset = datasets.ImageFolder(
        ROOT_DIR,
        transform = (get_train_transform(IMAGE_SIZE, pretrained))
    )

    dataset_test = datasets.ImageFolder(
        ROOT_DIR,
        transform= (get_valid_transform(IMAGE_SIZE, pretrained))
    )
    dataset_size = len(dataset)
    testset_size = int(VALID_SPLIT * dataset_size)
    #randomize the indices
    indices = torch.randperm(len(dataset)).tolist()

    #train and test sets:
    dataset_train = Subset(dataset, indices[: -testset_size])
    dataset_test = Subset(dataset_test, indices[-testset_size :])

    return dataset_train, dataset_test, dataset.classes

def get_data_loaders(dataset_train, dataset_test):
    """
    prepares the training and test data loaders.

    :param dataset_train: the training dataset
    :param dataset_test: the testing dataset

    returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train,
        batch_size= BATCH_SIZE,
        shuffle = True,
        num_workers= NUM_WORKERS
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size = BATCH_SIZE,
        shuffle= False,
        num_workers= NUM_WORKERS
    )
    return train_loader, test_loader

"""if __name__ == "__main__":
    # test the dataset loader
    train_set, valid_set, classes = get_datasets(pretrained=False)
    print(len(train_set), len(valid_set), classes)
"""