# %% Import libraries
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score, confusion_matrix, plot_confusion_matrix

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

# %% Preprocess features and labels
scaler = StandardScaler().fit(X_train)
encoder = LabelEncoder().fit(y_train)

X_train, y_train = scaler.transform(X_train), encoder.transform(y_train)
X_dev, y_dev = scaler.transform(X_dev), encoder.transform(y_dev)

# %%
pca = PCA()
pca.fit(X_train, y_train)

plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.title("Explained variance by principal components")
plt.show()

variance_20_pc = sum(pca.explained_variance_ratio_[:20])
"""
The first 20 PCs account for 51% of the variance.
"""

# %%
pca99 = PCA(n_components=.99)
pca99.fit(X_train, y_train)

plt.figure()
plt.plot(pca99.explained_variance_ratio_)
plt.show()
"""
These PCs account for 99% of the variance present in the train data set.
In order to represent 99% of the variance, we only need 1709 principal components."""

# %% Use the PCs accounting for 99% of variance for training and development
pca_train = pca99.transform(X_train)
pca_dev = pca99.transform(X_dev)

svm_clf = LinearSVC(C=0.0001)
svm_clf.fit(pca_train, y_train)

# %%
pred_train, pred_dev = svm_clf.predict(pca_train), svm_clf.predict(pca_dev)
train_acc = svm_clf.score(pca_train, y_train)
dev_acc = svm_clf.score(pca_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')
#cf_mat = confusion_matrix(y_dev, pred_dev, labels=encoder.classes_, normalize=True)

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""
train_acc = 0.91, dev_acc = 0.49
train_uar = 0.91, dev_uar = 0.49
"""
plot_confusion_matrix(svm_clf, pca_dev, y_dev, normalize='pred', values_format='.1f')
plt.show()

