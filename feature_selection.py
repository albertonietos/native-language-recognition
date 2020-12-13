# %% Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import opensmile
from tqdm import tqdm
import csv
from sklearn.feature_selection import chi2
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from ReliefF import ReliefF
# %% Load training and development sets
train_csv = "ComParE_train.csv"
train_df = pd.read_csv(train_csv).sample(frac=1)
train_X = train_df.loc[:, train_df.columns != 'L1']
train_y = train_df.loc[:, train_df.columns == 'L1']

devel_csv = "ComParE_devel.csv"
devel_df = pd.read_csv(devel_csv).sample(frac=1)
devel_X = devel_df.loc[:, devel_df.columns != 'L1']
devel_y = devel_df.loc[:, devel_df.columns == 'L1']
# %% Rank features (chi2)
rank_metric = 'p_val'

chi2_scores, pvals = chi2(preprocessing.MinMaxScaler().fit_transform(train_X),train_y)

ranking_df = pd.DataFrame({'feature': train_X.columns, 'chi2_score': chi2_scores, 'p_val': pvals})

# %% sort by metric
rank_metric = 'p_val'
ranking_df.sort_values(rank_metric, inplace=True)
ranking_df.head()
# %% 
k_feature_range = [30,50,100,200,300,400,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,len(train_X.columns)]
UAR_scores_train = []
UAR_scores_devel = []
for k_features in tqdm(k_feature_range):
    selected_features = ranking_df.feature[0:k_features]
    train_X_selected = train_X[train_X.columns.intersection(selected_features)]
    devel_X_selected = devel_X[devel_X.columns.intersection(selected_features)]

    clf_SVM = make_pipeline(preprocessing.StandardScaler(), SVC())
    clf_SVM.fit(train_X_selected,train_y)
    # train UAR
    train_pred = clf_SVM.predict(train_X_selected)
    UAR_train = recall_score(train_y,train_pred, average='macro')
    UAR_scores_train.append(UAR_train)
    # devel UAR
    devel_pred = clf_SVM.predict(devel_X_selected)
    UAR_devel = recall_score(devel_y,devel_pred, average='macro')
    UAR_scores_devel.append(UAR_devel)

# %% Plot figures
plt.figure(figsize=(20,10))
p_train = plt.plot(k_feature_range,UAR_scores_train, '-o', label="training set")
p_devel = plt.plot(k_feature_range,UAR_scores_devel, '-o', label="development set")

plt.xlabel("number of features")
plt.ylabel("UAR (%)")
plt.legend()
plt.title("number of best features VS. UAR for SVMs")
plt.show()
