"""
Configuration file for TinyImageNet100 Classification Project
All hyperparameters and paths are defined here for clean project structure.
"""

import torch

# =========================
# Device Configuration
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Dataset Configurationsource
# =========================
DATA_DIR = "data"
IMAGE_SIZE = 64
NUM_CLASSES = 100
BATCH_SIZE = 64
NUM_WORKERS = 0

# =========================
# Training Configuration
# =========================
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# =========================
# Model Saving
# =========================
MODEL_SAVE_PATH = "best_model.pth"

# =========================
# Early Stopping
# =========================
PATIENCE = 5