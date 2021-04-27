import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng()

###############################################################
### Parameters ###
lsym  = 1                      # length of symbol in samples
h_in  = 1                      # pulseshaping filter
Csize = [4]                    # size of constellation diagrams [4,4,8] means first sender has 4 bit, second sender 4 and third sender 8 bit
s_off = [0]                    # sample offset -> model imperfect synchronization

train_size = 20                # number of symbols in the training set
test_size = 10                 # number of symbols in the test data
batch_size = 5

# channel model is defined inside the nn

###############################################################

# check integrety of given values
if np.log2(Csize)==int(np.log2(Csize)):
    Bit_per_sym = int(np.log2(Csize))   # Bit per symbol for each sender
else:
    raise ValueError("Modulation formats need to be of size 2^n. Change Csize")

if np.size(Csize)!=np.size(s_off):
    raise ValueError("Csize and s_off should be of same size")

if train_size/batch_size!=int(train_size/batch_size):
    raise ValueError("train_size needs to be a multiple of the batch_size")


## creation of random message training set
train_message=rng.integers(0,2,(train_size,Bit_per_sym))
#print(train_message)
test_message=rng.integers(0,2,(test_size,Bit_per_sym))

# Create data loaders.
train_dataloader = DataLoader(train_message, batch_size=batch_size)
test_dataloader = DataLoader(test_message, batch_size=batch_size)

""" # Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!") """