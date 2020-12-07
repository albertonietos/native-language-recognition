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


# %% Models to try: Random Forests,

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)


# %% Evaluate model
pred_train, pred_dev = rf_clf.predict(X_train), rf_clf.predict(X_dev)
train_acc = rf_clf.score(X_train, y_train)
dev_acc = rf_clf.score(X_dev, y_dev)
train_uar = recall_score(y_train, pred_train, average='macro')
dev_uar = recall_score(y_dev, pred_dev, average='macro')

print(f"train_acc = {train_acc:.2f}, dev_acc = {dev_acc:.2f}")
print(f"train_uar = {train_uar:.2f}, dev_uar = {dev_uar:.2f}")

"""train_acc = 1.00, dev_acc = 0.31
train_uar = 1.00, dev_uar = 0.31"""

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

# %% Non-linear SVM
svc = SVC()
param_grid = [{'C': [0.0001, 0.001, 0.01]}]
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='recall_macro', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
# see results
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)

"""0.24878787878787878 {'C': 0.0001}
0.24878787878787878 {'C': 0.001}
0.24878787878787878 {'C': 0.01}"""

"""WORK ONLY WITH MFCC COEFFICIENTS"""

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