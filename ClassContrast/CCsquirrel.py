import pandas as pd
import os
import re
import time
import numpy as np
import statistics
import random
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from xgboost import XGBClassifier

import networkx as nx
from networkx import ego_graph

import torch.optim as optim
import argparse
import torch

#from logger import Logger
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.loader import DataLoader


def spatial_two(Node_class, Edge_indices, n):
    F_vec = []
    for i in range(n):
        #print("\rProcessing file {} ({}%)".format(i, 100 * i // (n - 1)), end='', flush=True)
        node_F = []
        list_out = []
        list_In = []
        S_nbd_out = []
        S_nbd_in = []
        for edge in Edge_indices:
            src, dst = edge
            if src == i:
                list_out.append(label[dst])
                for edge_2 in Edge_indices:
                    src_2, dst_2 = edge_2
                    if src_2 == dst and src_2 != dst_2:
                        S_nbd_out.append(label[dst_2])

        # print(list_out)
        # print(list_In)
        for d in Node_class:
            count = 0
            count_in = 0

            for node in list_out:
                if Node_class[node] == d:
                    count += 1
            node_F.append(count)

        for d in Node_class:
            count_S_out = 0
            count_S_in = 0
            for node in S_nbd_out:
                if Node_class[node] == d:
                    count_S_out += 1
            node_F.append(count_S_out)

        F_vec.append(node_F)
    return F_vec
def Similarity(array1, array2):
    intersection = np.sum(np.logical_and(array1, array2))
    return intersection


def Domain_Fe(DataFram, basis, sel_basis, feature_names):
    Fec = []
    for i in range(Number_nodes):
        # for i in range(2):
        vec = []
        f = DataFram.loc[i, feature_names].values.flatten().tolist()
        vec.append(Similarity(f, basis[0]))
        vec.append(Similarity(f, basis[1]))
        vec.append(Similarity(f, basis[2]))
        vec.append(Similarity(f, basis[3]))
        vec.append(Similarity(f, basis[4]))
        f.clear()
        Fec.append(vec)
    SFec = []
    for i in range(Number_nodes):
        # for i in range(2):
        Svec = []
        f = DataFram.loc[i, feature_names].values.flatten().tolist()
        Svec.append(Similarity(f, sel_basis[0]))
        Svec.append(Similarity(f, sel_basis[1]))
        Svec.append(Similarity(f, sel_basis[2]))
        Svec.append(Similarity(f, sel_basis[3]))
        Svec.append(Similarity(f, sel_basis[4]))
        f.clear()
        SFec.append(Svec)
    return Fec, SFec
def Result(result):
    #feature=[]
    feature=[]
    for i in range(len(x[0])-1):
        feature.append("{}".format(i))
    l=5
    for i in range(l):
        feature.append("S_{}".format(i))
    for i in range(l):
        feature.append("I_{}".format(i))



    X=result[feature] # Features
    y=result['Class']  # Labels
    X_train=X.iloc[Train]
    X_test=X.iloc[test_index]
    y_train=y.iloc[Train]
    y_test=y.iloc[test_index]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Don't cheat - fit only on training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(700,), random_state=1,max_iter=1000, warm_start=True)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%\n")
    #Both.append(metrics.accuracy_score(y_test, y_pred)*100)
    return metrics.accuracy_score(y_test, y_pred)*100
def Result_Spatial(result):
    #feature=[]
    feature=[]
    for i in range(len(x[0])):
        feature.append("{}".format(i))



    X=result[feature] # Features
    y=result['Class']  # Labels
    X_train=X.iloc[Train]
    X_test=X.iloc[test_index]
    y_train=y.iloc[Train]
    y_test=y.iloc[test_index]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Don't cheat - fit only on training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(700,), random_state=1,max_iter=1000, warm_start=True)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy using Spatial Feature only:",metrics.accuracy_score(y_test, y_pred)*100,"%\n")
    #Both.append(metrics.accuracy_score(y_test, y_pred)*100)
    return metrics.accuracy_score(y_test, y_pred)*100
def Result_Domain(result):
    feature=[]
    l=5
    for i in range(l):
        feature.append("S_{}".format(i))
    for i in range(l):
        feature.append("I_{}".format(i))



    X=result[feature] # Features
    y=result['Class']  # Labels
    X_train=X.iloc[Train]
    X_test=X.iloc[test_index]
    y_train=y.iloc[Train]
    y_test=y.iloc[test_index]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Don't cheat - fit only on training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # apply same transformation to test data
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(700,), random_state=1,max_iter=1000, warm_start=True)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy using Domain Feature:",metrics.accuracy_score(y_test, y_pred)*100,"%\n")
    #Both.append(metrics.accuracy_score(y_test, y_pred)*100)
    return metrics.accuracy_score(y_test, y_pred)*100

def ClassContrast(result):
    feature=[]
    for i in range(len(x[0])-1):
        feature.append("{}".format(i))
    l=5
    for i in range(l):
        feature.append("S_{}".format(i))
    for i in range(l):
        feature.append("I_{}".format(i))



    X=result[feature] # Features
    y=result['Class']  # Labels
    X_train=X.iloc[Train]
    X_test=X.iloc[test_index]
    y_train=y.iloc[Train]
    y_test=y.iloc[test_index]

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    # fit the model
    num_features_to_select = 15
    model.fit(X_train,y_train)
    weight=model.get_booster().get_score(importance_type='weight')
    sorted_dict = {k: v for k, v in sorted(weight.items(), key=lambda item: (-item[1], item[0]))}
    best_features = list(sorted_dict.keys())[:num_features_to_select]

    #train using Best feature
    X=result[best_features] # Features
    y=result['Class']  # Labels
    X_train=X.iloc[Train]
    X_test=X.iloc[test_index]
    y_train=y.iloc[Train]
    y_test=y.iloc[test_index]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Don't cheat - fit only on training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # apply same transformation to test data
    X_test = scaler.transform(X_test)
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(700,), random_state=1,max_iter=1000, warm_start=True)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100,"%\n")
    return metrics.accuracy_score(y_test, y_pred)*100

Accuracy = []
Accuracy_S = []
Accuracy_D = []
Accuracy_CC = []
dataset = WikipediaNetwork(root='/tmp/squirrel', name='squirrel')
data = dataset[0]
Number_nodes = len(data.y)
label = data.y.numpy()
Edge_idx = data.edge_index.numpy()
Node = range(Number_nodes)
Edgelist = []
for i in range(len(Edge_idx[1])):
    Edgelist.append((Edge_idx[0][i], Edge_idx[1][i]))
# print(Edgelist)
Node_class = [0, 1, 2, 3, 4, 5]
for run in range(10):
    Domain_Fec = pd.DataFrame(data.x.numpy())
    label = pd.DataFrame(data.y.numpy(), columns=['class'])
    Data = pd.concat([Domain_Fec, label], axis=1)
    Data.head()
    label = data.y.numpy()

    Number_nodes = len(data.y)
    fe_len = len(data.x[0])
    catagories = Data['class'].to_numpy()
    data_by_class = {cls: Data.loc[Data['class'] == cls].drop(['class'], axis=1) for cls in range(max(catagories) + 1)}
    basis = [[max(df[i]) for i in range(len(df.columns))] for df in data_by_class.values()]
    sel_basis = [[int(list(df[i].to_numpy()).count(1) >= int(len(df[i].index) * 0.01))
                  for i in range(len(df.columns))]
                 for df in data_by_class.values()]
    dataset = WikipediaNetwork(root='/tmp/squirrel', name='squirrel')
    data = dataset[0]
    feature_names = [ii for ii in range(fe_len)]
    idx_train = [data.train_mask[i][run] for i in range(len(data.y))]
    train_index = np.where(idx_train)[0]
    idx_val = [data.val_mask[i][run] for i in range(len(data.y))]
    valid_index = np.where(idx_val)[0]
    idx_test = [data.test_mask[i][run] for i in range(len(data.y))]
    test_index = np.where(idx_test)[0]
    # num_class=np.max(label)
    for idx_test in test_index:
        label[idx_test] = 5

    Train = np.concatenate((train_index, valid_index))
    print('Run= ', run)
    F_vec = spatial_two(Node_class, Edgelist, Number_nodes)
    # print(F_vec)
    x = np.array(F_vec)
    k = len(F_vec[0])
    feature = []
    for i in range(len(x[0])):
        feature.append("{}".format(i))
    data_s = pd.DataFrame(x, columns=feature)
    data_s.insert(loc=k, column='Class', value=data.y)
    data_s.head()
    Fec, SFec = Domain_Fe(Data, basis, sel_basis, feature_names)
    Fe = []
    for i in range(5):
        Fe.append("S_{}".format(i))
    Z = np.array(SFec)
    data_2 = pd.DataFrame(Z, columns=Fe)
    data_2.head()
    Fe = []
    for i in range(5):
        Fe.append("I_{}".format(i))
    y = np.array(Fec)
    data1 = pd.DataFrame(y, columns=Fe)
    data1.head()
    result = pd.concat([data1, data_2, data_s], axis=1)
    result.head()
    acc_CC=ClassContrast(result)
    Accuracy_CC.append(acc_CC)
    accuracy = Result(result)
    acc_Spa = Result_Spatial(result)
    acc_dom = Result_Domain(result)
    Accuracy.append(accuracy)
    Accuracy_S.append(acc_Spa)
    Accuracy_D.append(acc_dom)
print(statistics.mean(Accuracy_CC))
print(statistics.stdev(Accuracy_CC))
print(statistics.mean(Accuracy_S))
print(statistics.stdev(Accuracy_S))
print(statistics.mean(Accuracy_D))
print(statistics.stdev(Accuracy_D))
print(statistics.mean(Accuracy))
print(statistics.stdev(Accuracy))

