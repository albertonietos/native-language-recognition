# %% Import libraries
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC

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

# %% Filter out any feature that is not from MFCCs
X_train, X_dev = X_train.filter(regex='mfcc*'), X_dev.filter(regex='mfcc*')

# %% Preprocess features and labels
scaler = StandardScaler().fit(X_train)
encoder = LabelEncoder().fit(y_train)

X_train, y_train = scaler.transform(X_train), encoder.transform(y_train)
X_dev, y_dev = scaler.transform(X_dev), encoder.transform(y_dev)

# %% Linear SVM model
svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)

pred_train, pred_dev = svm_clf.predict(X_train), svm_clf.predict(X_dev)
train_acc = svm_clf.score(X_train, y_train)
dev_acc = svm_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""train_acc = 1.00, dev_acc = 0.30
train_uar = 1.00, dev_uar = 0.30"""

# %% Linear SVM fine-tuning
svm_clf = LinearSVC()
param_grid = [{'C': [0.0001, 0.00001, 0.000001]}]
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)
# see results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

"""0.22575757575757577 {'C': 1e-06}
0.31696969696969696 {'C': 1e-05}
0.37424242424242427 {'C': 0.0001}
0.3693939393939394 {'C': 0.001}
0.32212121212121214 {'C': 0.01}"""

#%% Use best linear SVM in dev set
svm_clf = grid_search.best_estimator_
pred_train, pred_dev = svm_clf.predict(X_train), svm_clf.predict(X_dev)
train_acc = svm_clf.score(X_train, y_train)
dev_acc = svm_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

""""
train_acc = 0.65, dev_acc = 0.39
train_uar = 0.65, dev_uar = 0.39

The best estimator reaches .39 recall on the dev set.
"""

# %% Random forest
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X_train, y_train)
pred_train, pred_dev = rf.predict(X_train), rf.predict(X_dev)
train_acc = rf.score(X_train, y_train)
dev_acc = rf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""
train_acc = 1.00, dev_acc = 0.25
train_uar = 1.00, dev_uar = 0.26
"""
# %% Logistic regression
log_clf = LogisticRegression(penalty='l2', n_jobs=-1)
param_grid = [{'C': [0.0001, 0.001, 0.01]}]
grid_search = GridSearchCV(log_clf, param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1, refit=True)
grid_search.fit(X_train, y_train)

# see results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

"""
0.2887878787878788 {'C': 0.0001}
0.373030303030303 {'C': 0.001}
0.3657575757575758 {'C': 0.01}
"""

# %% K-Neighbors
neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
neigh.fit(X_train, y_train)
pred_train, pred_dev = neigh.predict(X_train), neigh.predict(X_dev)
train_acc = neigh.score(X_train, y_train)
dev_acc = neigh.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""
train_acc = 0.47, dev_acc = 0.20
train_uar = 0.47, dev_uar = 0.21
"""