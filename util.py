import os
import gzip
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
class DealDataset():
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name) 
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath:
        y_data = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x_data = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_data), 28, 28)
    return (x_data, y_data)

def get_data(data_dir='./data', batch_size=64):
    """
    Downloads and loads the MNIST dataset using PyTorch's torchvision library.

    Args:
        data_dir (str): Directory where the MNIST data will be stored.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for testing dataset.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0, variance 1
    ])
    
    # Download and load MNIST datasets
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
