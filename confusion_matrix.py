import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.dataset import get_dataloaders
from src.model import SimpleCNN
import config

# Load data
train_loader, test_loader, class_names = get_dataloaders()

# Load model
model = SimpleCNN().to(config.DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=config.DEVICE))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(config.DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("assets/confusion_matrix.png")

plt.show()