# %% Import libraries
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# %% Import data
folder_path = "C:/native_language/data/"
train = pd.read_csv(os.path.join(folder_path + "train.csv"))
dev = pd.read_csv(os.path.join(folder_path+"devel.csv"))

# %% Format data
# Separate into features (X) and labels (y)
X_train, y_train = train.iloc[:, 1:], train.loc[:, ["l1_nationality"]]
X_dev, y_dev = dev.iloc[:, 1:], dev.loc[:, ["l1_nationality"]]

# delete unneeded data
del train, dev

# %% Build pipeline
# feat_pipeline = Pipeline('std_scaler', StandardScaler())
# label_pipeline = Pipeline(['label_encoder', LabelEncoder()])
scaler = StandardScaler().fit(X_train)
#encoder = OneHotEncoder(sparse=False).fit(y_train)
encoder = LabelEncoder().fit(y_train)

X_train, y_train = scaler.transform(X_train), encoder.transform(y_train)
X_dev, y_dev = scaler.transform(X_dev), encoder.transform(y_dev)

# %% Train SVM model
svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)

# %% Evaluate model
pred_train, pred_dev = svm_clf.predict(X_train), svm_clf.predict(X_dev)
train_acc = svm_clf.score(X_train, y_train)
dev_acc = svm_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""train_acc = 1.00, dev_acc = 0.38
train_uar = 1.00, dev_uar = 0.38"""

# %% Fine tune the regularization parameter
# This is done in the baseline paper with the development set
"""param_grid = [{'C': [0.1, 1.0, 10, 100, 1000]}]
svm_clf = LinearSVC()
grid_search = GridSearchCV(svm_clf, param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)"""

# %%
"""print(grid_search.cv_results_)

plt.plot(grid_search.cv_results_['mean_test_score'])
plt.show()"""

# %%
"""param_grid2 = [{'C': [0.001, 0.01]}]
svm_clf2 = LinearSVC()
grid_search2 = GridSearchCV(svm_clf2, param_grid2, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search2.fit(X_train, y_train)"""

# %%
"""print(grid_search2.cv_results_)  # 0.41030303, 0.36545455])

plt.plot(grid_search2.cv_results_['mean_test_score'])
plt.show()"""

# %%
"""param_grid2 = [{'C': [0.00001, 0.0001]}]
svm_clf2 = LinearSVC()
grid_search2 = GridSearchCV(svm_clf2, param_grid2, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search2.fit(X_train, y_train)

# %%
print(grid_search2.cv_results_)  # [0.41333333, 0.45818182]

plt.plot(grid_search2.cv_results_['mean_test_score'])
plt.show()"""

# %%
param_grid2 = [{'C': [0.0001, 0.0001*3, 0.0001/3]}]  # [0.45818182, 0.44212121, 0.45212121]
svm_clf2 = LinearSVC()
grid_search2 = GridSearchCV(svm_clf2, param_grid2, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search2.fit(X_train, y_train)

# %%
print(grid_search2.cv_results_)  # [0.41333333, 0.45818182]

plt.plot(grid_search2.cv_results_['mean_test_score'])
plt.show()

# %% Final model
svm_clf = LinearSVC(C=0.0001)
svm_clf.fit(X_train, y_train)

pred_train, pred_dev = svm_clf.predict(X_train), svm_clf.predict(X_dev)
train_acc = svm_clf.score(X_train, y_train)
dev_acc = svm_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""train_acc = 0.93, dev_acc = 0.50
train_uar = 0.93, dev_uar = 0.51"""