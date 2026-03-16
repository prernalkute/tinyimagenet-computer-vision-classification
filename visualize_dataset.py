import torch
import matplotlib.pyplot as plt
import random
from src.dataset import get_dataloaders
import math

def visualize_random_samples(num_images=9, save_path="assets/dataset_samples.png"):
    """
    Visualize a specific number of random training images from TinyImageNet100.

    Args:
        num_images: total number of images to display
        save_path: path to save the figure
    """

    train_loader, _, _ = get_dataloaders()
    dataset = train_loader.dataset.dataset  # underlying ImageFolder

    # Get class names
    class_names = dataset.classes

    # Randomly sample images
    indices = random.sample(range(len(dataset)), num_images)
    images, labels = zip(*[dataset[i] for i in indices])
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Determine grid size (rows x cols)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))

    # Denormalization values
    mean = torch.tensor([0.4802, 0.4481, 0.3975])
    std = torch.tensor([0.2302, 0.2265, 0.2262])

    axes = axes.flatten()  # flatten in case of multiple rows/cols
    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]
            label = labels[i].item()

            # Denormalize
            img = img * std[:, None, None] + mean[:, None, None]
            img = img.permute(1, 2, 0).numpy()
            img = img.clip(0, 1)

            # Plot
            ax.imshow(img, interpolation="nearest")
            ax.set_title(class_names[label], fontsize=8)
        ax.axis("off")  # Hide axes for all cells

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved dataset visualization to {save_path}")
    plt.show()


if __name__ == "__main__":
    visualize_random_samples(num_images=9)