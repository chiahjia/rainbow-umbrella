import numpy as np
import torch
import torch.nn as nn
import torch.functional as f
import torch.optim as optim

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


net = FaceEx(100, 100, 100, 100, 3)
print(net)
opt = optim.Adam(net.parameters(), lr=0.1)
loss_func = nn.NLLLoss()
dataset = []

for epoch in range(EPOCHS):
    for data in dataset:
        X, y = data
        net.zero_grad()
        output = net.forward(X)
        loss = loss_func(output)
        loss.backward()
        opt.step()
