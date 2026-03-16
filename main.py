"""
Main entry point for TinyImageNet100 Classification Project
Connects dataset, model, and training pipeline.
"""

import torch
import config
from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.train import evaluate_test, train_model


def main():
    print("Starting TinyImageNet100 Training Pipeline")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders()
    print("Data loaded successfully.")

    # Initialize Model
    model = SimpleCNN().to(config.DEVICE)
    print(f"Using device: {config.DEVICE}")

    # Train Model
    model = train_model(model, train_loader, val_loader)

    # Evaluate on Test Set
    evaluate_test(model, test_loader)


if __name__ == "__main__":
    main()