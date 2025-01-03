import os
import gzip
import numpy as np
import torch
from torchvision import datasets, transforms

class DealDataset():
    """
    Custom dataset class for handling local MNIST data.

    Args:
        images (numpy.ndarray): Image data.
        labels (numpy.ndarray): Label data.
        transform (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.images[index], int(self.labels[index])
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)

def convert_to_numpy(dataset):
    """
    Converts PyTorch dataset to numpy arrays for images and labels.

    Args:
        dataset (torchvision.datasets): The dataset to be converted.

    Returns:
        tuple: Tuple of numpy arrays (images, labels).
    """
    images = dataset.data.numpy()
    labels = dataset.targets.numpy()
    return images, labels

def get_data(data_dir='./data'):
    """
    Downloads MNIST dataset and prepares it for use with DealDataset.

    Args:
        data_dir (str): Directory to store the MNIST data.

    Returns:
        tuple: Training and testing datasets as DealDataset instances.
    """
    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0, variance 1
    ])

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=None)

    # Convert PyTorch dataset to numpy arrays
    train_images, train_labels = convert_to_numpy(train_dataset)
    test_images, test_labels = convert_to_numpy(test_dataset)

    # Wrap in DealDataset
    train_data = DealDataset(train_images, train_labels, transform=transform)
    test_data = DealDataset(test_images, test_labels, transform=transform)

    return train_data, test_data
