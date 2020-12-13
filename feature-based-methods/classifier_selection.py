# %% Import libraries
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

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


# %% Random Forest

rf_clf = RandomForestClassifier(max_depth=30, min_samples_split=5, n_estimators=1000) # after fine-tuning
rf_clf.fit(X_train, y_train)


# %% Evaluate model
pred_train, pred_dev = rf_clf.predict(X_train), rf_clf.predict(X_dev)
train_acc = rf_clf.score(X_train, y_train)
dev_acc = rf_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

""" Default parameters
train_acc = 1.00, dev_acc = 0.31
train_uar = 1.00, dev_uar = 0.31"""

"""
train_acc = 1.00, dev_acc = 0.37
train_uar = 1.00, dev_uar = 0.37
"""

# %% Fine-tune RF
param_grid = [{'n_estimators': [100, 500, 1000],
               'max_features': ["auto"],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 5, 10],
               'max_depth': [5, 10, 30]}]
rf_clf = RandomForestClassifier()
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# %% see results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

"""0.2639393939393939 {'max_features': 'auto', 'n_estimators': 50}
0.28757575757575754 {'max_features': 'auto', 'n_estimators': 100}
0.3187878787878788 {'max_features': 'auto', 'n_estimators': 500}
0.2533333333333333 {'max_features': 'sqrt', 'n_estimators': 50}
0.28909090909090907 {'max_features': 'sqrt', 'n_estimators': 100}
0.3109090909090909 {'max_features': 'sqrt', 'n_estimators': 500}
0.2390909090909091 {'max_features': 'log2', 'n_estimators': 50}
0.2748484848484849 {'max_features': 'log2', 'n_estimators': 100}
0.2996969696969697 {'max_features': 'log2', 'n_estimators': 500}"""

"""0.27 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
0.27181818181818185 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
0.2778787878787879 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}
0.2709090909090909 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
0.27878787878787875 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}
0.27939393939393936 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1000}
0.26878787878787874 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}
0.27454545454545454 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 500}
0.276969696969697 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 1000}
0.26090909090909087 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 100}
0.27121212121212124 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 500}
0.2766666666666667 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}
0.2675757575757576 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 100}
0.2833333333333333 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 500}
0.2787878787878788 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 1000}
0.2718181818181818 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}
0.27575757575757576 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 500}
0.28090909090909094 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 1000}
0.26969696969696966 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 100}
0.28030303030303033 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 500}
0.2742424242424243 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 1000}
0.26999999999999996 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 100}
0.2787878787878788 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 500}
0.2790909090909091 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 1000}
0.2736363636363637 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 100}
0.27303030303030307 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 500}
0.27999999999999997 {'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 1000}
0.29909090909090913 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
0.3039393939393939 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
0.3103030303030303 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}
0.29060606060606065 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
0.31 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}
0.30727272727272725 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1000}
0.2921212121212121 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}
0.30636363636363634 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 500}
0.3103030303030303 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 1000}
0.2924242424242424 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 100}
0.3063636363636364 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 500}
0.3096969696969697 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}
0.29454545454545455 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 100}
0.30727272727272725 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 500}
0.3124242424242424 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 1000}
0.2921212121212121 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}
0.3054545454545455 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 500}
0.31242424242424244 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 1000}
0.29121212121212114 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 100}
0.3024242424242424 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 500}
0.3127272727272727 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 1000}
0.2951515151515152 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 100}
0.3051515151515152 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 500}
0.306969696969697 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 1000}
0.29242424242424236 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 100}
0.30999999999999994 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 500}
0.3103030303030303 {'max_depth': 10, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 1000}
0.28121212121212125 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
0.31212121212121213 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
0.3184848484848485 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1000}
0.29545454545454547 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
0.31787878787878787 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}
0.3196969696969697 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 1000}
0.296969696969697 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}
0.31606060606060604 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 500}
0.31636363636363635 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 1000}
0.2924242424242424 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 100}
0.3030303030303031 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 500}
0.31727272727272726 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}
0.29030303030303034 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 100}
0.31424242424242427 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 500}
0.31606060606060604 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 1000}
0.29393939393939394 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 100}
0.3163636363636364 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 500}
0.31666666666666665 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 1000}
0.2936363636363636 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 100}
0.30363636363636365 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 500}
0.3090909090909091 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 1000}
0.29666666666666663 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 100}
0.3036363636363636 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 500}
0.303030303030303 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 1000}
0.2948484848484848 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 100}
0.3051515151515151 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 500}
0.3124242424242424 {'max_depth': 30, 'max_features': 'auto', 'min_samples_leaf': 10, 'min_samples_split': 10, 'n_estimators': 1000}
grid_search.best_estimator_
Out[15]: RandomForestClassifier(max_depth=30, min_samples_split=5, n_estimators=1000)
0.32
"""

# %% LASSO, Ridge and elastic net
# smaller C, stronger regularization
"""param_grid = [{'penalty': ['l1'],
               'C': [0.01, 0.1, 1.],
               'solver': ['liblinear']},
              {'penalty': ['l2'],
               'C': [0.0001, 0.001, 0.01],
               'solver': ['lbfgs']},
              {'penalty': ['elasticnet'],
               'C': [0.0001, 0.001, 0.01],
               'l1_ratio': [0.3, 0.6, 0.9],
               'solver': ['saga']}]"""

param_grid = [{'penalty': ['elasticnet'],
               'C': [0.0001, 0.001, 0.01],
               'l1_ratio': [0.3, 0.6, 0.9],
               'solver': ['saga']}]
param_grid = [{'penalty': ['l1'],
               'C': [0.0001, 0.001, 0.01, 0.1, 1.],
               'solver': ['liblinear']}]
log_clf = LogisticRegression(n_jobs=-1)
grid_search = GridSearchCV(log_clf, param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# see results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

"""0.2866666666666667 {'C': 1e-05}
0.380909090909091 {'C': 0.0001}
0.45181818181818184 {'C': 0.001}
0.4442424242424242 {'C': 0.01}
0.4372727272727273 {'C': 0.1}
0.4333333333333334 {'C': 1.0}"""

"""0.09090909090909091 {'C': 0.0001, 'penalty': 'l1'}
0.37909090909090915 {'C': 0.0001, 'penalty': 'l2'}
0.09090909090909091 {'C': 0.001, 'penalty': 'l1'}
0.4412121212121212 {'C': 0.001, 'penalty': 'l2'}
0.22121212121212125 {'C': 0.01, 'penalty': 'l1'}
0.44363636363636366 {'C': 0.01, 'penalty': 'l2'}
0.09090909090909091 {'C': 0.0001, 'l1_ratio': 0.3, 'penalty': 'elasticnet'}
0.09090909090909091 {'C': 0.0001, 'l1_ratio': 0.6, 'penalty': 'elasticnet'}
0.09090909090909091 {'C': 0.0001, 'l1_ratio': 0.9, 'penalty': 'elasticnet'}
0.09090909090909091 {'C': 0.001, 'l1_ratio': 0.3, 'penalty': 'elasticnet'}
0.09090909090909091 {'C': 0.001, 'l1_ratio': 0.6, 'penalty': 'elasticnet'}
0.09090909090909091 {'C': 0.001, 'l1_ratio': 0.9, 'penalty': 'elasticnet'}
0.3687878787878788 {'C': 0.01, 'l1_ratio': 0.3, 'penalty': 'elasticnet'}
0.29272727272727267 {'C': 0.01, 'l1_ratio': 0.6, 'penalty': 'elasticnet'}
0.23242424242424242 {'C': 0.01, 'l1_ratio': 0.9, 'penalty': 'elasticnet'}
"""

"""0.09090909090909091 {'C': 0.0001, 'l1_ratio': 0.3, 'penalty': 'elasticnet', 'solver': 'saga'}
0.09090909090909091 {'C': 0.0001, 'l1_ratio': 0.6, 'penalty': 'elasticnet', 'solver': 'saga'}
0.09090909090909091 {'C': 0.0001, 'l1_ratio': 0.9, 'penalty': 'elasticnet', 'solver': 'saga'}
0.09090909090909091 {'C': 0.001, 'l1_ratio': 0.3, 'penalty': 'elasticnet', 'solver': 'saga'}
0.09090909090909091 {'C': 0.001, 'l1_ratio': 0.6, 'penalty': 'elasticnet', 'solver': 'saga'}
0.09090909090909091 {'C': 0.001, 'l1_ratio': 0.9, 'penalty': 'elasticnet', 'solver': 'saga'}
0.36939393939393933 {'C': 0.01, 'l1_ratio': 0.3, 'penalty': 'elasticnet', 'solver': 'saga'}
0.29363636363636364 {'C': 0.01, 'l1_ratio': 0.6, 'penalty': 'elasticnet', 'solver': 'saga'}
0.23363636363636364 {'C': 0.01, 'l1_ratio': 0.9, 'penalty': 'elasticnet', 'solver': 'saga'}"""

"""
0.09090909090909091 {'C': 0.0001, 'penalty': 'l1', 'solver': 'liblinear'}
0.09090909090909091 {'C': 0.001, 'penalty': 'l1', 'solver': 'liblinear'}
0.2315151515151515 {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'}
0.4333333333333333 {'C': 0.1, 'penalty': 'l1', 'solver': 'liblinear'}
0.40181818181818185 {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear'}
"""
# %% Fine-tuned logistic regression - L2
log_clf = LogisticRegression(penalty='l2', C=0.001, max_iter=1000)
log_clf.fit(X_train, y_train)

pred_train, pred_dev = log_clf.predict(X_train), log_clf.predict(X_dev)
train_acc = log_clf.score(X_train, y_train)
dev_acc = log_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

# %% Fine-tuned logistic regression - L1
log_clf = LogisticRegression(penalty='l1', C=0.1, max_iter=1000, solver='liblinear')
log_clf.fit(X_train, y_train)

pred_train, pred_dev = log_clf.predict(X_train), log_clf.predict(X_dev)
train_acc = log_clf.score(X_train, y_train)
dev_acc = log_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""
train_acc = 0.94, dev_acc = 0.44
train_uar = 0.94, dev_uar = 0.44
"""

# %% Fine-tuned logistic regression - ElasticNet
log_clf = LogisticRegression(penalty='elasticnet', C=0.01, max_iter=1000, solver='saga', l1_ratio=0.3)
log_clf.fit(X_train, y_train)

pred_train, pred_dev = log_clf.predict(X_train), log_clf.predict(X_dev)
train_acc = log_clf.score(X_train, y_train)
dev_acc = log_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

# %% Non-linear SVM
svc = SVC(kernel='rbf')
param_grid = [{'C': [0.001, 0.01, 0.1],
               'gamma': [0.001, 0.01, 0.1]}]
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
# see results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

"""0.24878787878787878 {'C': 0.0001}
0.24878787878787878 {'C': 0.001}
0.24878787878787878 {'C': 0.01}"""

"""
0.17181818181818181 {'C': 0.001, 'gamma': 0.001}
0.1996969696969697 {'C': 0.001, 'gamma': 0.01}
0.1778787878787879 {'C': 0.001, 'gamma': 0.1}
0.17181818181818181 {'C': 0.01, 'gamma': 0.001}
0.1996969696969697 {'C': 0.01, 'gamma': 0.01}
0.17818181818181816 {'C': 0.01, 'gamma': 0.1}
0.17181818181818181 {'C': 0.1, 'gamma': 0.001}
0.1996969696969697 {'C': 0.1, 'gamma': 0.01}
0.1787878787878788 {'C': 0.1, 'gamma': 0.1}
"""

# %%
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)

pred_train, pred_dev = svc.predict(X_train), svc.predict(X_dev)
train_acc = svc.score(X_train, y_train)
dev_acc = svc.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

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
train_acc = 0.49, dev_acc = 0.25
train_uar = 0.49, dev_uar = 0.26
"""

# %% Fine-tuning
neigh = KNeighborsClassifier()
param_grid = [{'n_neighbors': [5, 15, 30],
               'p': [1, 2]}]
grid_search = GridSearchCV(neigh, param_grid, scoring='recall_macro', n_jobs=-1, refit=True, cv=5, verbose=2)
grid_search.fit(X_train, y_train)
# see results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

"""
0.10030303030303031 {'n_neighbors': 5, 'p': 1}
0.10242424242424245 {'n_neighbors': 5, 'p': 2}
0.12121212121212119 {'n_neighbors': 15, 'p': 1}
0.11787878787878787 {'n_neighbors': 15, 'p': 2}
0.11424242424242426 {'n_neighbors': 30, 'p': 1}
0.11363636363636362 {'n_neighbors': 30, 'p': 2}
"""
# %%
ada_clf = AdaBoostClassifier(n_estimators=100)
ada_clf.fit(X_train, y_train)

pred_train, pred_dev = ada_clf.predict(X_train), ada_clf.predict(X_dev)
train_acc = ada_clf.score(X_train, y_train)
dev_acc = ada_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""train_acc = 0.29, dev_acc = 0.24
train_uar = 0.29, dev_uar = 0.25"""

# %%
gboost_clf = GradientBoostingClassifier(n_estimators=100)
gboost_clf.fit(X_train, y_train)

pred_train, pred_dev = gboost_clf.predict(X_train), gboost_clf.predict(X_dev)
train_acc = gboost_clf.score(X_train, y_train)
dev_acc = gboost_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""train_acc = 0.99, dev_acc = 0.37
train_uar = 0.99, dev_uar = 0.37
very long training time"""

# %% Naive Bayes
bayes_clf = GaussianNB()
bayes_clf.fit(X_train, y_train)

pred_train, pred_dev = bayes_clf.predict(X_train), bayes_clf.predict(X_dev)
train_acc = bayes_clf.score(X_train, y_train)
dev_acc = bayes_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""
train_acc = 0.28, dev_acc = 0.23
train_uar = 0.28, dev_uar = 0.24
"""

