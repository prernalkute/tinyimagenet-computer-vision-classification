import matplotlib.pyplot as plt

# Example training values (replace with your real values if saved)
train_loss = [3.49,2.33,2.13,1.97,1.84,1.72,1.65,1.57,1.50,1.41,1.38,1.35,1.26,1.24,1.16,1.11,1.03,0.99,0.97,0.93]
train_acc = [14,21,29,34,39,42,44,47,49,52,53,53,57,58,60,62,64,65,65,68]
val_acc = [24,31,38,42,42,46,40,50,49,50,49,47,56,49,42,53,57,53,55,60]

epochs = range(1,21)

plt.figure(figsize=(10,5))

plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig("assets/training_accuracy.png")

plt.show()