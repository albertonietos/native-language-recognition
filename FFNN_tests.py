# %% Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score

# %% Import data
folder_path = "C:/native_language/data/"
train = pd.read_csv(os.path.join(folder_path + "train.csv"))
dev = pd.read_csv(os.path.join(folder_path + "devel.csv"))

# %% Format data
# Separate into features (X) and labels (y)
X_train, y_train = train.iloc[:, 1:], train["l1_nationality"]
X_dev, y_dev = dev.iloc[:, 1:], dev["l1_nationality"]

# delete unneeded data
del train, dev

# %% Build pipeline
scaler = StandardScaler().fit(X_train)
encoder = LabelEncoder().fit(y_train)

X_train, y_train = scaler.transform(X_train), encoder.transform(y_train)
X_dev, y_dev = scaler.transform(X_dev), encoder.transform(y_dev)

# %% Build MLP
device = torch.device("cuda:0")
n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))

X_train, y_train = torch.Tensor(X_train), torch.from_numpy(y_train)
X_dev, y_dev = torch.Tensor(X_dev), torch.from_numpy(y_dev)


# %%
class MLP(nn.Module):
    def __init__(self, n_inputs=n_inputs, n_outputs=n_outputs):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def compute_uar(model, inputs, targets):
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward(inputs)
        uar = recall_score(targets.cpu().numpy(), np.argmax(outputs.cpu().numpy(), axis=1),
                           average='macro')

        return uar


l2_lambda = 0.001


def train_model(X_train, y_train, X_dev, y_dev, l2_lambda=l2_lambda):
    model = MLP()
    model.to(device)

    train_uar = []
    dev_uar = []
    X_train, y_train = X_train.to(device), y_train.to(device, dtype=torch.int64)
    X_dev, y_dev = X_dev.to(device), y_dev.to(device, dtype=torch.int64)
    opt = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for i in range(200):
        model.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        # Apply regularization
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + l2_lambda * l2_norm
        loss.backward()
        if i % 10:
            trn, dev = compute_uar(model, X_train, y_train), compute_uar(model, X_dev, y_dev)
            train_uar.append(trn)
            dev_uar.append(dev)
        opt.step()

    return train_uar, dev_uar


# %%
train_uar, dev_uar = train_model(X_train, y_train, X_dev, y_dev, l2_lambda=l2_lambda)

plt.plot(train_uar, label='train')
plt.plot(dev_uar, label='dev')
plt.title("Evolution of MLP training")
plt.legend()
plt.show()

# %% Grid Search L2 lambda
lmbd = [0.000001, 0.00001, 0.0001, 0.001, 0.01]
lmbd = [0.001, 0.001*3, 0.001*9]
train_list, dev_list = [], []
for l2 in lmbd:
    train_uar, dev_uar = train_model(X_train, y_train, X_dev, y_dev, l2_lambda=l2)
    train_list.append(train_uar[-1])
    dev_list.append(dev_uar[-1])
    print(f"l2={l2}, train_uar={train_uar[-1]:.2f}, dev_uar={dev_uar[-1]:.2f}")

"""l2=1e-06, train_uar=1.00, dev_uar=0.45
l2=1e-05, train_uar=1.00, dev_uar=0.47
l2=0.0001, train_uar=1.00, dev_uar=0.48
l2=0.001, train_uar=1.00, dev_uar=0.46
l2=0.01, train_uar=0.33, dev_uar=0.32"""
