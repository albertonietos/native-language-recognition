# %% Import libraries
import os

import pandas as pd
from sklearn.metrics import recall_score, plot_confusion_matrix, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from mlp_utils import train_model

# %% Import data
folder_path = "C:/native_language/data/"
train = pd.read_csv(os.path.join(folder_path,"train.csv"))
dev = pd.read_csv(os.path.join(folder_path,"devel.csv"))
test = pd.read_csv(os.path.join(folder_path, "test.csv"))

# %% Concatenate train and development data
train = pd.concat([train, dev])

# %% Format data
# Separate into features (X) and labels (y)
X_train, y_train = train.iloc[:, 1:], train["l1_nationality"]
X_test, y_test = test.iloc[:, 1:], test["L1"].values

# delete unneeded data
del train, dev, test

# %% Build pipeline
scaler = StandardScaler().fit(X_train)
encoder = LabelEncoder().fit(y_train)

X_train, y_train = scaler.transform(X_train), encoder.transform(y_train)
X_test = scaler.transform(X_test)

# test labels are already encoded

# %% Fit SVM model with best hyperparameters
svm_clf = LinearSVC(C=0.0001)
svm_clf.fit(X_train, y_train)

# %% Compute final metrics for SVM
pred_train, pred_dev = svm_clf.predict(X_train), svm_clf.predict(X_test)
train_acc = svm_clf.score(X_train, y_train)
test_acc = svm_clf.score(X_test, y_test)
train_uar = recall_score(y_train, pred_train, average='macro')
test_uar = recall_score(y_test, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, test_acc = {test_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, test_uar = {test_uar:.3f}")

"""
train_acc = 0.91, test_acc = 0.51
train_uar = 0.91, test_uar = 0.51
"""

plt.figure()
disp = plot_confusion_matrix(svm_clf, X_test, y_test,
                             display_labels=encoder.classes_,
                             cmap=plt.cm.Blues)
disp.ax_.set_title('Confusion matrix')
plt.savefig('confusion_mat_test_SVM.png', dpi=300)

# %% Fit MLP with best hyperparameters
device = torch.device("cuda:0")

X_train, y_train = torch.Tensor(X_train), torch.from_numpy(y_train)
X_test, y_test = torch.Tensor(X_test), torch.from_numpy(y_test)

model, train_uar, test_uar = train_model(X_train, y_train,
                                 X_test, y_test,
                                 l2_lambda=0.001,
                                 lr=0.001)

print(f"train_uar={max(train_uar):.3f}, "
      f"dev_uar={max(test_uar):.3f}") # train_uar=0.992, dev_uar=0.528
# %% Compute final metrics for MLP
plt.figure()
plt.plot(train_uar, label='train')
plt.plot(test_uar, label='test')
plt.title("Evolution of MLP training")
plt.legend()
plt.show()
plt.close()

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

cf = confusion_mat(model, X_test, y_test)
plt.figure()
sns.heatmap(cf, annot=True, xticklabels=encoder.classes_,
                yticklabels=encoder.classes_, fmt='d',
            cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.savefig('confusion_mat_test_FFNN.png', dpi=300)
plt.show()




