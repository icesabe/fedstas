import torch
from torch.utils.data import DataLoader

def evaluate_model(model, dataset, batch_size=64, device="cpu"):
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    return accuracy