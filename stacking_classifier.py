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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier

# %% Import data
folder_path = "C:/native_language/data/"
train = pd.read_csv(os.path.join(folder_path + "train.csv"))
dev = pd.read_csv(os.path.join(folder_path+"devel.csv"))

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

# %%
estimators = [
    ('svm', LinearSVC(C=0.0001)),
    ('log', LogisticRegression(penalty='l2', C=0.001, max_iter=1000))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=GradientBoostingClassifier()
)

clf.fit(X_train, y_train)

pred_train, pred_dev = clf.predict(X_train), clf.predict(X_dev)
train_acc = clf.score(X_train, y_train)
dev_acc = clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""
train_acc = 0.83, dev_acc = 0.47
train_uar = 0.83, dev_uar = 0.47
"""