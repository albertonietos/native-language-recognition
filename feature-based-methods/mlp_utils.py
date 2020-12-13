# %% Import libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import recall_score, confusion_matrix

# %% Define cuda device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


# %% Define MLP model
class MLP(nn.Module):
    def __init__(self, h_size=200, n_inputs=6373, n_outputs=11):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_inputs, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_outputs)

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


def train_model(X_train, y_train, X_dev, y_dev, model=MLP(h_size=200),
                l2_lambda=0.001, lr=0.001):
    model.to(device)

    train_uar = []
    dev_uar = []
    X_train, y_train = X_train.to(device), y_train.to(device, dtype=torch.int64)
    X_dev, y_dev = X_dev.to(device), y_dev.to(device, dtype=torch.int64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for i in range(500):
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

    return model, train_uar, dev_uar


def predict(net, X, y):
    net.eval()
    pred_lst = []
    true_lst = []
    with torch.no_grad():
        x, y = X.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
        out = net(x)
        _, pred = torch.max(out.data, 1)
        pred_lst.extend(pred.cpu())
        true_lst.extend(y.cpu())
    return pred_lst, true_lst


def confusion_mat(net, X, y, normalize=False):
    pred_lst, true_lst = predict(net, X, y)
    cf = confusion_matrix(true_lst, pred_lst)
    if normalize:
        cf = cf / cf.sum(axis=1, keepdims=True)

    return cf
