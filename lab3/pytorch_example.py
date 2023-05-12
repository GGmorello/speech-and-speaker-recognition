
# This file contains boiler-plate code for defining and training a network in PyTorch.
# Please see PyTorch documentation and tutorials for more information 
# e.g. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import editdistance
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc3=nn.Linear(256,256)
        self.fc4=nn.Linear(256,256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc2(x)
        return x

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def load_data():
    lmfcc_train_x = np.load('lab3/d_mspec_train.npz')['d_mspec_train']
    lmfcc_val_x = np.load('lab3/d_mspec_val.npz')['d_mspec_val']
    lmfcc_test_x = np.load('lab3/d_mspec_test.npz')['d_mspec_test']
    one_hot_train_y = np.load('lab3/one_hot_train_y.npz')['one_hot_train_y']
    one_hot_val_y = np.load('lab3/one_hot_val_y.npz')['one_hot_val_y']
    one_hot_test_y = np.load('lab3/one_hot_test_y.npz')['one_hot_test_y']
    return lmfcc_train_x, lmfcc_val_x, lmfcc_test_x, one_hot_train_y, one_hot_val_y, one_hot_test_y

def transform_sequence(sequence):
    new=[]
    for i in range(len(sequence)):
        if i==0:
            new.append(sequence[i])
        elif sequence[i]!=sequence[i-1]:
            new.append(sequence[i])
    return new

 
def calculate_PER(reference, predicted):
    reference_length = len(reference)
    edit_dist = editdistance.eval(reference, predicted)
    PER = edit_dist / reference_length
    return PER
train_x, val_x, test_x, train_y, val_y, test_y = load_data()
# Convert string labels to numerical format

output_dim = train_y.shape[1]

# # Instantiate the network and print the structure
# net = Net(train_x.shape[1], output_dim).to(device)
# print(net)
# print('Number of parameters:', count_parameters(net))

# # Define the loss criterion
criterion = nn.CrossEntropyLoss()

# # Define the optimizer
# optimizer = torch.optim.Adam(net.parameters())

batch_size = 256

# # Create the data loaders for training and validation sets
# train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y).float())
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# val_dataset = torch.utils.data.TensorDataset(torch.Tensor(val_x), torch.Tensor(val_y).float())
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# # Setup logging for TensorBoard
# #writer = SummaryWriter()

# # Train the network
# num_epochs = 10
# train_losses=[]
# val_losses=[]
# val_accuracies=[]

# for epoch in range(num_epochs):
#     net.train()
#     train_loss = 0.0
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     net.eval()
#     with torch.no_grad():
#         val_loss = 0.0
#         correct=0
#         for inputs, labels in val_loader:
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             #for each datapoint, find the index of the max value in the output vector
#             _, predicted = torch.max(outputs, 1)
#             #for each row in labels find where the value is 1
#             _, labels = torch.max(labels, 1)
#             #count the number of correct predictions
#             correct += (predicted == labels).sum().item()
#             val_loss += loss.item()
           
#     val_accuracy = correct / len(val_loader.dataset)
#     train_loss /= len(train_loader)
#     val_loss /= len(val_loader)
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
#     val_accuracies.append(val_accuracy)
#     if epoch>=2 and val_loss>val_losses[-1] and val_loss>val_losses[-2]:
#         break
#     print(f'Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}', f'val_accuracy={val_accuracy}')

# #plot the training and validation loss
# import matplotlib.pyplot as plt
# plt.plot(train_losses, label='Training loss')
# plt.plot(val_losses, label='Validation loss')
# plt.legend()
# plt.show()
# #plot the validation accuracy
# plt.plot(val_accuracies, label='Validation accuracy')
# plt.legend()
# plt.show()


# #save the model
# torch.save(net.state_dict(), 'lab3/d_mspec_4hidden.pt')

# # Evaluate the model on the test set
# net.eval()

net = Net(train_x.shape[1], output_dim).to(device)
net.load_state_dict(torch.load('lab3/d_mspec_3hidden.pt'))
net.eval()
stateList = np.load('lab3/statelist.npz',allow_pickle=True)['arr_0']
test_loss = 0.0
correct_state = 0
correct_phe=0
total = 0
test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y).float())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

reference=[]
reference_pheno=[]
test_y_pheno=np.zeros((test_y.shape[0],len(stateList)))
for i in range(test_y.shape[0]):
    #find the state index that is 1
    state_index=np.where(test_y[i]==1)[0][0]
    name_state=stateList[state_index]
    reference.append(name_state)
    name_state=name_state.split('_')[0]
    reference_pheno.append(name_state)

    #find the index of the phenome
    indexes=[]
    for j in range(len(stateList)):
        if stateList[j].split('_')[0]==name_state:
            indexes.append(j)
    #make the one hot vector
    for j in indexes:
        test_y_pheno[i][j]=1


print(reference[:10],"stooop",reference_pheno[:10])






prediction=[]
prediction_pheno=[]
with torch.no_grad():
    batch=0
    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        for i in range(predicted.shape[0]):
            prediction.append(stateList[predicted[i]])
            prediction_pheno.append(stateList[predicted[i]].split('_')[0])
            if test_y_pheno[i+batch*batch_size][predicted[i]]==1:
                correct_phe+=1 
        
        total += labels.size(0)
        test_loss+=loss.item()
        #for each row in labels find where the value is 1
        _, labels = torch.max(labels, 1)
        #count the number of correct predictions
        correct_state += (predicted == labels).sum().item()
        batch+=1

print(prediction[:10],"stooop",prediction_pheno[:10])

reference=transform_sequence(reference)
reference_pheno=transform_sequence(reference_pheno)
prediction=transform_sequence(prediction)
prediction_pheno=transform_sequence(prediction_pheno)

print(reference[:10],"stooop",reference_pheno[:10])
print(prediction[:10],"stooop",prediction_pheno[:10])




test_loss /= len(test_loader)
test_accuracy_state = correct_state / total
test_accuracy_pheno=correct_phe/total
PER1=calculate_PER(reference,prediction)
PER2=calculate_PER(reference_pheno,prediction_pheno)
state_cm = confusion_matrix(reference, prediction, labels=stateList)
phenomenon_cm = confusion_matrix(reference_pheno, prediction_pheno, labels=stateList)

print("State Confusion Matrix:")
print(state_cm)
print("Phenomenon Confusion Matrix:")
print(phenomenon_cm)


print(f'Test Loss: {test_loss}, Test Accuracy state: {test_accuracy_state}', f'Test Accuracy Phenomene: {test_accuracy_pheno}', f'PER: {PER1}', f'PER: {PER2}')

