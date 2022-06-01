# %% ANCHOR
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# %% ANCHOR
# Download training data from open datasets
training_data = datasets.FashionMNIST(
    root      = "data",
    train     = True,
    download  = True,
    transform = ToTensor()
)

# Download test data from open datasets
test_data = datasets.FashionMNIST(
    root      = "data",
    train     = False,
    download  = True,
    transform = ToTensor()
)

# %% ANCHOR
batch_size = 64

# Create data loaders
train_data_loader = DataLoader(training_data, batch_size=batch_size)
test_data_loader  = DataLoader(test_data, batch_size=batch_size)

for X, y in test_data_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# %% ANCHOR
# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        