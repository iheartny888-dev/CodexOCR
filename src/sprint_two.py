import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

epochs = 4
convpool = 1
channels = 28
fc = 3

train_data = datasets.EMNIST(root="./datasets", split="letters", train=True, download=True, transform=transforms.ToTensor(), target_transform=lambda y: y - 1)
test_data = datasets.EMNIST(root="./datasets", split="letters", train=False, download=True, transform=transforms.ToTensor(), target_transform=lambda y: y - 1)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class Model(nn.Module) :
    def __init__(self) :
        super().__init__()

        convpool_layers = []
        in_channels = 1
        dim = 28

        for i in range(convpool) :
            out_channels = (channels // convpool) * (i + 1)
            convpool_layers.extend([nn.Conv2d(in_channels, out_channels, 3, 1), nn.ReLU(), nn.MaxPool2d(2, 2)])
            dim = (dim - 2) // 2
            in_channels = out_channels

        self.convpool = nn.Sequential(*convpool_layers)
        flattened = dim * dim * in_channels

        fc_layers = []
        in_features = flattened

        for i in range(fc - 1) :
            out_features = max(1, int(flattened * (2/3) ** (i + 1)))
            fc_layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
            in_features = out_features

        fc_layers.append(nn.Linear(in_features, 26))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x) :
        x = self.convpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

torch.manual_seed(14)
model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()

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

        if b % 320 == 0 :
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
print(f'Time: {total_time/60} minutes')

test_accuracy = (test_correct[-1].item() / len(test_data)) * 100
print(f'Accuracy: {test_accuracy:.2f}%')