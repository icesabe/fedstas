import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import SGD
from typing import Dict, Any

from client.sampling import sample_uniform_data

def local_train(
        model: Module,
        dataset: torch.utils.data.Dataset,
        epochs: int,
        batch_size: int,
        lr: float,
        sample_fraction: float,
        device: str = "cpu"
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

    Returns:
        Module: Updated local model after training
    """
    model = model.to(device)
    model.train()

    # Step 1: Sample uniformly from the local dataset
    subset = sample_uniform_data(dataset, sample_fraction)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    # Step 2: Set up optimizer and loss
    optimizer = SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss() # We can parametrize this

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