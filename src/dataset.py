"""
Dataset loading and preprocessing for TinyImageNet100
Handles training, validation and test data preparation.
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config


def get_transforms(train=True):

    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(config.IMAGE_SIZE, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4802, 0.4481, 0.3975],
                std=[0.2302, 0.2265, 0.2262]
            )
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4802, 0.4481, 0.3975],
                std=[0.2302, 0.2265, 0.2262]
            )
        ])


def get_dataloaders():

    train_path = os.path.join(config.DATA_DIR, "train")
    test_path = os.path.join(config.DATA_DIR, "test")

    full_train_dataset = datasets.ImageFolder(
        root=train_path,
        transform=get_transforms(train=True)
    )

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size]
    )

    # Validation should not use augmentation
    val_dataset.dataset.transform = get_transforms(train=False)

    test_dataset = datasets.ImageFolder(
        root=test_path,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, test_loader