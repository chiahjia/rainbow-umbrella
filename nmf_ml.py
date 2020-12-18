import numpy as np
import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
import detect_face
#import dnmf
#import gabor
from torch.utils.data import Dataset, DataLoader
import os

expressions = {'N': 0,
               'A': 1,
               'F': 2,
               'C': 3, #HC
               'O': 4 #HO
               }


nmf_data = []
nmf_labels = []
path = os.getcwd() + '/nmf_dest_occluded'
folders = os.listdir(path)

for fold in folders:
    temp_path = path + '/' + str(fold)
    files = os.listdir(temp_path)

    for file in files:

        if str(file)[-3:] == 'txt':
            text = open(temp_path + '/' + str(file))
            lst = text.read().splitlines()
            data_tens = [float(i) for i in lst]
            label = expressions[str(file)[-5]]
            nmf_data.append(data_tens)
            nmf_labels.append(label)


class NMF_dataset(Dataset):

    def __init__(self, data, labels):

        self.data = data
        self.labels = labels

    def __getitem__(self, index):

        dat = torch.tensor(self.data[index]).float()
        lab = torch.tensor(self.labels[index])

        return (dat, lab)

    def __len__(self):

        return len(self.data)


dataset = NMF_dataset(nmf_data, nmf_labels)

train_set, test_set = torch.utils.data.random_split(dataset, [1100, 107])

trainloader = DataLoader(train_set, batch_size=10, shuffle=True)
testloader = DataLoader(test_set, batch_size=1, shuffle=True)



class NmfDetect(nn.Module):

    def __init__(self, input, h1, h2, classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.lsm = nn.Softmax()

        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, classes)

    def forward(self, batch):
        out = self.fc1(batch)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


net = NmfDetect(300, 60, 60, 5)

opt = optim.Adam(net.parameters(), lr=0.1)
loss_func = nn.CrossEntropyLoss(torch.tensor([0.2, 1, 1, 1, 1]))

EPOCHS = 5

for epoch in range(EPOCHS):
    for data in trainloader:
        X, y = data

        net.zero_grad()
        output = net(X)
        loss = loss_func(output, y)
        #print(loss)
        loss.backward()
        opt.step()


test_correct = 0
total = 0
for data in testloader:
    X, y = data
    output = net(X)
    if torch.argmax(output) == y[0]:
        #print(torch.argmax(output))
        test_correct += 1
    total += 1

print(test_correct / total)
