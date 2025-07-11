### Run in google colab
import os
import sys
import importlib

# Step 1: Go to base dir and pull latest code
os.chdir('/content')
REPO_URL = "https://github.com/JaSlesso/fedstas.git"
REPO_DIR = "fedstas"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL}
else:
    !git -C {REPO_DIR} fetch origin
    !git -C {REPO_DIR} reset --hard origin/HEAD
    !git -C {REPO_DIR} clean -fd

# Step 2: Drop cached modules
for module in list(sys.modules):
    if module.startswith("server.") or module.startswith("client."):
        del sys.modules[module]

# Step 3: Enter the repo
os.chdir(REPO_DIR)


#Above is the code to run in google colab

### This is the code to run test for CIFAR-10

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt

from server.coordinator import FedSTaSCoordinator
from model.simple_cifar10 import create_model


# -----------------------
# Configuration
# -----------------------
config = {
    "H": 5,
    "d_prime": 10,           # slightly higher for CIFAR-10 complexity
    "restratify_every": 5,
    "clients_per_round": 10,
    "M": 500,               # higher bound for CIFAR-10 datasets
    "epsilon": 8.0,         # slightly lower for more privacy
    "alpha": (np.exp(8.0) - 1) / (np.exp(8.0) + 499),  # adjusted for new M
    "n_star": 500,          # higher target for CIFAR-10
    "epochs": 5,            # more epochs for CIFAR-10
    "batch_size": 64,       # larger batch size
    "lr": 0.001,
    "weight_decay": 1e-4,   # slightly higher regularization
    "verbose": True
}


# -----------------------
# Dataset: Dirichlet Split for CIFAR-10
# -----------------------
def split_cifar10_dirichlet(dataset, num_clients=20, beta=0.5):
    """Split CIFAR-10 dataset using Dirichlet distribution for non-IID data"""
    labels = np.array([target for _, target in dataset])
    indices = np.arange(len(dataset))
    class_indices = [indices[labels == i] for i in range(10)]  # 10 classes in CIFAR-10

    client_indices = defaultdict(list)
    for c in range(10):
        proportions = np.random.dirichlet(np.repeat(beta, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        split = np.split(class_indices[c], proportions)
        for i, idx in enumerate(split):
            client_indices[i].extend(idx.tolist())

    return [Subset(dataset, sorted(idxs)) for i, idxs in sorted(client_indices.items())]


# CIFAR-10 specific transforms
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 datasets
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Create non-IID client datasets
client_datasets = split_cifar10_dirichlet(cifar10_train, num_clients=20, beta=0.5)

print(f"Created {len(client_datasets)} client datasets")
print(f"Dataset sizes: {[len(ds) for ds in client_datasets[:5]]}...")  # Show first 5


# -----------------------
# Model and Coordinator
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Choose model type: "simple_cnn" or "resnet18"
MODEL_TYPE = "simple_cnn"  # Change to "resnet18" for ResNet-18
global_model = create_model(MODEL_TYPE, num_classes=10)

print(f"Using {MODEL_TYPE} model")
print(f"Model parameters: {sum(p.numel() for p in global_model.parameters() if p.requires_grad):,}")

coordinator = FedSTaSCoordinator(
    global_model=global_model,
    client_datasets=client_datasets,
    test_dataset=cifar10_test,
    config=config,
    device=device,
    verbose=True
)

# -----------------------
# Run Simulation
# -----------------------
print("Starting FedSTaS training on CIFAR-10...")
coordinator.run(num_rounds=25)  # More rounds for CIFAR-10


# -----------------------
# Plot Results
# -----------------------
plt.figure(figsize=(15, 5))

# Plot validation accuracy
plt.subplot(1, 3, 1)
plt.plot(coordinator.validation_curve, 'b-', linewidth=2)
plt.title(f"CIFAR-10 Validation Accuracy ({MODEL_TYPE})")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

# Plot validation loss
plt.subplot(1, 3, 2)
plt.plot(coordinator.validation_loss_curve, 'r-', linewidth=2)
plt.title(f"CIFAR-10 Validation Loss ({MODEL_TYPE})")
plt.xlabel("Round")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True, alpha=0.3)

# Plot dataset distribution (first 10 clients)
plt.subplot(1, 3, 3)
client_sizes = [len(ds) for ds in client_datasets[:10]]
plt.bar(range(10), client_sizes)
plt.title("Dataset Sizes (First 10 Clients)")
plt.xlabel("Client Index")
plt.ylabel("Number of Samples")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final results
if coordinator.validation_curve:
    final_accuracy = coordinator.validation_curve[-1]
    best_accuracy = max(coordinator.validation_curve)
    print(f"\nFinal Results:")
    print(f"Final Accuracy: {final_accuracy*100:.2f}%")
    print(f"Best Accuracy: {best_accuracy*100:.2f}%")
    print(f"Model: {MODEL_TYPE}")
    print(f"Total Rounds: {len(coordinator.validation_curve)}")
