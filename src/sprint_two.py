import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plot
import time

train_data = datasets.EMNIST(root="./datasets/EMNIST", split="letters", train=True, download=True, transform=transforms.ToTensor(), target_transform=lambda y: y - 1)
test_data = datasets.EMNIST(root="./datasets/EMNIST", split="letters", train=False, download=True, transform=transforms.ToTensor(), target_transform=lambda y: y - 1)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

class Model(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 26)

    def forward(self, x) :
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(X, dim=1)

torch.manual_seed(14)
model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

start_time = time.time()

epochs = 4
train_loss = []
test_loss = []
train_correct = []
test_correct = []

for i in range(epochs) :
    tr_c = 0
    ts_c = 0

    for b, (x_train, y_train) in enumerate(train_loader) :
        b += 1
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_correct = (predicted == y_train).sum()
        tr_c += batch_correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 500 == 0 :
            print(f'Epoch: {i} Batch: {b} Loss: {loss.item()}')

    train_loss.append(loss)
    train_correct.append(tr_c)

    with torch.no_grad() :
        for b, (x_test, y_test) in enumerate(test_loader) :
            b += 1
            y_val = model(x_test)
            predicted = torch.max(y_val.data, 1)[1]
            ts_c += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_loss.append(loss)
    test_correct.append(ts_c)

current_time = time.time()
total_time = current_time - start_time
print(f'Training time: {total_time/60} minutes')