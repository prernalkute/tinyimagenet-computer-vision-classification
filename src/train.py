"""
Training and Evaluation Pipeline for TinyImageNet100 Classification

This script handles:
1. Model training
2. Validation monitoring
3. Early stopping
4. Model checkpoint saving
5. Final testing
6. Performance visualization
7. Confusion matrix generation
8. Classification report

These steps provide a complete evaluation of the CNN model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import config


def train_model(model, train_loader, val_loader):
    """
    Trains the CNN model using the training dataset and evaluates
    performance on the validation dataset after every epoch.

    Early stopping is applied to prevent overfitting.
    The best performing model is saved.
    """

    # ===============================
    # Loss Function
    # ===============================
    # CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # ===============================
    # Optimizer
    # ===============================
    # Adam optimizer helps achieve faster convergence
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # ===============================
    # Training History (for graphs)
    # ===============================
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    # ===============================
    # Early Stopping Parameters
    # ===============================
    best_val_accuracy = 0
    patience_counter = 0

    # ===============================
    # Training Loop
    # ===============================
    for epoch in range(config.NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        # Iterate over training batches
        for images, labels in tqdm(train_loader):

            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

            # Compute predictions
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        # ===============================
        # Training Metrics
        # ===============================
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # ===============================
        # Validation Phase
        # ===============================
        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)

                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        val_accuracies.append(val_accuracy)

        # ===============================
        # Early Stopping Check
        # ===============================
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            # Save best model
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

            print("Model improved. Saved.")

            patience_counter = 0

        else:

            patience_counter += 1

            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:

            print("Early stopping triggered.")

            break

    print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")

    # ===============================
    # Plot Training Graphs
    # ===============================
    plot_training_graphs(train_losses, train_accuracies, val_accuracies)

    return model


# =========================================================
# Training Graph Visualization
# =========================================================
def plot_training_graphs(train_losses, train_accuracies, val_accuracies):

    epochs = range(1, len(train_losses) + 1)

    plt.figure()

    plt.plot(epochs, train_losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

    plt.figure()

    plt.plot(epochs, train_accuracies)
    plt.plot(epochs, val_accuracies)

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend(["Train Accuracy", "Validation Accuracy"])

    plt.show()


# =========================================================
# Test Evaluation
# =========================================================
def evaluate_test(model, test_loader):

    """
    Evaluates the trained model on the unseen test dataset.
    Also generates confusion matrix and classification report.
    """

    model.eval()

    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct / total

    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

    # ===============================
    # Confusion Matrix
    # ===============================
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10,8))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.show()

    # ===============================
    # Classification Report
    # ===============================
    print("\nClassification Report")

    print(classification_report(all_labels, all_predictions))

    return test_accuracy