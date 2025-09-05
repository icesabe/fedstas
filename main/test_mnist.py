### Run in google colab
import os
import sys
import importlib
'''
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

'''
#Above is the code to run in google colab





### This is the code to run test for MNIST

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt

from server.coordinator import FedSTaSCoordinator
from model.simple_mnist import SimpleCNN


# -----------------------
# Configuration
# -----------------------
config = {
    "H": 5,
    "d_prime": 5,
    "restratify_every": 5,  # slightly more frequent
    "clients_per_round": 10,
    "M": 500,
    "epsilon": 10.0,
    "alpha": (np.exp(10.0) - 1) / (np.exp(10.0) + 99),
    "n_star": 200,          # reduce to trigger subsampling
    "epochs": 3,            # longer local updates
    "batch_size": 32,
    "lr": 0.001,            # reduce learning rate
    "weight_decay": 1e-5,   # add to Adam (update train loop)
    "verbose": True
}


# -----------------------
# Dataset: Dirichlet Split
# -----------------------
def split_mnist_dirichlet(dataset, num_clients=20, beta=0.5):
    labels = np.array([target for _, target in dataset])
    indices = np.arange(len(dataset))
    class_indices = [indices[labels == i] for i in range(10)]

    client_indices = defaultdict(list)
    for c in range(10):
        proportions = np.random.dirichlet(np.repeat(beta, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        split = np.split(class_indices[c], proportions)
        for i, idx in enumerate(split):
            client_indices[i].extend(idx.tolist())

    return [Subset(dataset, sorted(idxs)) for i, idxs in sorted(client_indices.items())]


transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

client_datasets = split_mnist_dirichlet(mnist_train, num_clients=20, beta=0.5)


# -----------------------
# Model and Coordinator
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

global_model = SimpleCNN()

coordinator = FedSTaSCoordinator(
    global_model=global_model,
    client_datasets=client_datasets,
    test_dataset=mnist_test,
    config=config,
    device=device,
    verbose=True
)

# -----------------------
# Run Simulation
# -----------------------
coordinator.run(num_rounds=20)


# -----------------------
# Plot Results
# -----------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(coordinator.validation_curve)
plt.title("Validation Accuracy")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(coordinator.validation_loss_curve)
plt.title("Validation Loss")
plt.xlabel("Round")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)

plt.tight_layout()
plt.show()



