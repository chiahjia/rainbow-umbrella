import numpy as np
import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim
import detect_face
#import dnmf
#import gabor
import os
import random

expressions = {'N': [1, 0, 0, 0, 0],
               'A': [0, 1, 0, 0, 0],
               'F': [0, 0, 1, 0, 0],
               'C': [0, 0, 0, 1, 0], #HC
               'O': [0, 0, 0, 0, 1] #HO
               }

gabor_data = []
gabor_labels = []
path = os.getcwd() + '/gabor_output_txt/gabors'
folders = os.listdir(path)
print(folders)
for fold in folders:
    temp_path = path + '/' + str(fold)
    files = os.listdir(temp_path)

    for file in files:

        if str(file)[-3:] == 'txt':
            text = open(temp_path + '/' + str(file))
            lst = text.read().splitlines()
            data_tens = torch.Tensor([int(i) for i in lst])

            label = torch.Tensor(expressions[str(file)[-5]])
            gabor_data.append(data_tens)
            gabor_labels.append(label)

            # gabor_dataset.append(thing)

# make batches:
data_batches = []
label_batches = []
length = len(gabor_labels)
while len(gabor_data) > 0:
    i = 0
    batch = []
    labels = []
    while i < 5:
        index = random.randint(0, length - 1)
        batch.append(gabor_data[index])
        labels.append(gabor_labels[index])
        gabor_data.pop(index)
        gabor_labels.pop(index)
        length -= 1
        i += 1
    data_batches.append(batch)

    label_batches.append(labels)

data_batches = torch.Tensor(data_batches)
label_batches = torch.Tensor(label_batches)

EPOCHS = 3
class FaceEx(nn.Module):

    def __init__(self, input, h1, h2, h3, classes):
        super().__init__()
        self.relu = nn.ReLU()
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
        return nn.LogSoftmax(self.fc4(out))


net = FaceEx(1680, 60, 60, 60, 5)

opt = optim.Adam(net.parameters(), lr=0.1)
loss_func = nn.NLLLoss()
dataset = []

for epoch in range(EPOCHS):
    for i in range(len(data_batches)):

        net.zero_grad()
        output = net(data_batches[i])
        loss = loss_func(output, label_batches[i])
        loss.backward()
        opt.step()
