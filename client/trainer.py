import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Adam
from typing import Dict, Any

from client.sampling import sample_uniform_data

def local_train(
        model: Module,
        dataset: torch.utils.data.Dataset,
        epochs: int,
        batch_size: int,
        lr: float,
        sample_fraction: float,
        device: str = "cpu",
        loss_fn = None
) -> Module:
    """
    Perform local training on a uniformly sampled subset of the dataset.

    Args:
        model (Module): The global model sent by the server
        dataset (Dataset): Full local dataset for the client
        epochs (int): Number of local training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate for optimizer
        sample_fraction (float): Fraction of local data to train on (from server)
        device (str): 'cpu' or 'cuda'
        loss_fn (callable, optional): Loss function (defaults to CrossEntropyLoss)

    Returns:
        Module: Updated local model after training
    """
    model = model.to(device)
    model.train()

    # Step 1: Sample uniformly from the local dataset
    subset = sample_uniform_data(dataset, sample_fraction)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    # Step 2: Set up optimizer and loss
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()

    # Step 3: Train
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
    return model


def get_model_gradient(old_model: torch.nn.Module, new_model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten and return the model difference (pseudo-gradient) as a single 1D tensor.

    Args:
        old_model (Module): The original global model before training
        new_model (Module): The locally updated model

    Returns:
        Tensor: Flattened gradient-like vector (1D tensor)
    """
    diffs = []
    for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
        diffs.append((new_param.data - old_param.data).flatten())
    return torch.cat(diffs)

def get_raw_update(model, dataset, device="cpu"):
    """
    Compute raw gradient vector from one batch of data.
    """
    model = model.to(device)
    model.train()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    model.zero_grad()
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten().detach().clone())
        else:
            grads.append(torch.zeros_like(p.data.flatten()))
    return torch.cat(grads).cpu().numpy()