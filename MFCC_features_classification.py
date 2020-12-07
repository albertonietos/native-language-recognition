# %% Import libraries
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# %% Import data
folder_path = "C:/native_language/data/"
train = pd.read_csv(os.path.join(folder_path + "train.csv"))
dev = pd.read_csv(os.path.join(folder_path + "devel.csv"))

# %% Format data
# Separate into features (X) and labels (y)
X_train, y_train = train.iloc[:, 1:], train.loc[:, ["l1_nationality"]]
X_dev, y_dev = dev.iloc[:, 1:], dev.loc[:, ["l1_nationality"]]

# delete unneeded data
del train, dev

# %% Filter out any feature that is not from MFCCs
X_train, X_dev = X_train.filter(regex='mfcc*'), X_dev.filter(regex='mfcc*')

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

# %%
svm_clf = LinearSVC()
param_grid = [{'C': [0.0001, 0.001, 0.01]}]
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
# see results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)


"""0.29545454545454547 {'C': 0.0001}
0.22606060606060607 {'C': 0.001}
0.16757575757575757 {'C': 0.01}"""

#%%
svm_clf = grid_search.best_estimator_
pred_train, pred_dev = svm_clf.predict(X_train), svm_clf.predict(X_dev)
train_acc = svm_clf.score(X_train, y_train)
dev_acc = svm_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

""""train_acc = 0.53, dev_acc = 0.33
train_uar = 0.53, dev_uar = 0.33

The best estimator (trained on 4/5 of training data due to refit=False in grid_search)
reaches .33 recall on the dev set."""