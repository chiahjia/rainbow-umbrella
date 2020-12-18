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
import random

# expressions = {'N': [1, 0, 0, 0, 0],
#                'A': [0, 1, 0, 0, 0],
#                'F': [0, 0, 1, 0, 0],
#                'C': [0, 0, 0, 1, 0], #HC
#                'O': [0, 0, 0, 0, 1] #HO
#                }

expressions = {'N': 0,
               'A': 1,
               'F': 2,
               'C': 3, #HC
               'O': 4 #HO
               }

n = 0
a = 0
gabor_data = []
gabor_labels = []
path = os.getcwd() + '/gabor_output_2/gabors'
folders = os.listdir(path)
print(folders)
for fold in folders:
    temp_path = path + '/' + str(fold)
    files = os.listdir(temp_path)

    for file in files:

        if str(file)[-3:] == 'txt':
            text = open(temp_path + '/' + str(file))
            lst = text.read().splitlines()
            data_tens = [int(i) for i in lst]
            label = expressions[str(file)[-5]]
            if str(file)[-5] == 'A':
                a += 1
            if str(file)[-5] == 'N':
                n += 1
            gabor_data.append(data_tens)
            gabor_labels.append(label)


class GaborData(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        dat = torch.tensor(self.data[index]).float()
        lab = torch.tensor(self.labels[index])

        return (dat, lab)

    def __len__(self):
        return len(self.data)

dataset = GaborData(gabor_data, gabor_labels)
train_set, test_set = torch.utils.data.random_split(dataset, [1200, 95])

trainloader = DataLoader(train_set, batch_size=10, shuffle=True)
testloader = DataLoader(test_set, batch_size=5, shuffle=True)




EPOCHS = 3

class FaceEx(nn.Module):

    def __init__(self, input, h1, h2, h3, classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.lsm = nn.LogSoftmax()
        self.fc1 = nn.Linear(input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, classes)

    def forward(self, batch):
        out = self.fc1(batch)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        #return torch.tensor(np.linalg.norm(self.fc4(out).detach().numpy()))
        #return self.sig(self.fc4(out))
        return self.fc4(out)


net = FaceEx(1680, 100, 150, 150, 5)
#net.train()
opt = optim.Adam(net.parameters(), lr=0.1)
loss_func = nn.CrossEntropyLoss(torch.tensor([0.2, 1, 1, 1, 1]))



for epoch in range(EPOCHS):
    for data in trainloader:
        X, y = data

        net.zero_grad()
        output = net(X)
        print(output)
        loss = loss_func(output, y)

        loss.backward()
        opt.step()

print(net(test_set[0][0]), test_set[0][1])