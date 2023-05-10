
# This file contains boiler-plate code for defining and training a network in PyTorch.
# Please see PyTorch documentation and tutorials for more information 
# e.g. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder

class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2= nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def load_data():
    lmfcc_train_x = np.load('lab3/d_lmfcc_train.npz')['d_lmfcc_train']
    lmfcc_val_x = np.load('lab3/d_lmfcc_val.npz')['d_lmfcc_val']
    lmfcc_test_x = np.load('lab3/d_lmfcc_test.npz')['d_lmfcc_test']
    one_hot_train_y = np.load('lab3/one_hot_train_y.npz')['one_hot_train_y']
    one_hot_val_y = np.load('lab3/one_hot_val_y.npz')['one_hot_val_y']
    one_hot_test_y = np.load('lab3/one_hot_test_y.npz')['one_hot_test_y']
    return lmfcc_train_x, lmfcc_val_x, lmfcc_test_x, one_hot_train_y, one_hot_val_y, one_hot_test_y

train_x, val_x, test_x, train_y, val_y, test_y = load_data()

# Convert string labels to numerical format

output_dim = train_y.shape[1]

# Instantiate the network and print the structure
net = Net(train_x.shape[1], output_dim)
print(net)
print('Number of parameters:', count_parameters(net))

# Define the loss criterion
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(net.parameters())

batch_size = 128

# Create the data loaders for training and validation sets
train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y).float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y).float())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Setup logging for TensorBoard
#writer = SummaryWriter()

# Train the network
num_epochs = 100

for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        correct=0
        for inputs, labels in val_loader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #for each datapoint, find the index of the max value in the output vector
            _, predicted = torch.max(outputs, 1)
            #for each row in labels find where the value is 1
            _, labels = torch.max(labels, 1)
            #count the number of correct predictions
            correct += (predicted == labels).sum().item()
            val_loss += loss.item()
           
    val_accuracy = correct / len(val_loader.dataset)
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f'Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}', f'val_accuracy={val_accuracy}')


#save the model
torch.save(net.state_dict(), 'lab3/pytorch_model.pt')

# Evaluate the model on the test set
net.eval()

test_loss = 0.0
correct = 0
total = 0
test_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_x), torch.Tensor(test_y).float())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = correct / total

print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')