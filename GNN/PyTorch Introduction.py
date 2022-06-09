# %% ANCHOR Import libraries
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# %% ANCHOR Download training data from open datasets
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

# %% ANCHOR Visualizing data
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}
figure = plt.figure(figsize=(8, 8))
cols, rows  = 3, 3
for i in range(1, cols * rows + 1): 
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %% ANCHOR Batching data
# For reshuffling the data at every epoch to reduce model overfitting, 
# and use Python's multiprocessing to speed up data retrieval.
batch_size = 64

# Create data loaders
train_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_data_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_data_loader: 
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

