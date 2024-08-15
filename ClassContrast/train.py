import pandas as pd
import argparse
import statistics
from models import *
from dataloader import *
from torch_geometric.datasets import Planetoid
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric")
# Parameter change to generate output
runs=10
dataset_name='cornell' # cora,texas,cornell, wisconsin
Feature_ratio=0.37 #wis,cor=0.37,tx=0.46,cora=0.84,
#end parameter

Accuracy_CC = []
dataset=load_data(dataset_name)
data = dataset[0]
Number_nodes = len(data.y)
label = data.y.numpy()
Edge_idx = data.edge_index.numpy()
Node = range(Number_nodes)
Edgelist = []
for i in range(len(Edge_idx[1])):
    Edgelist.append((Edge_idx[0][i], Edge_idx[1][i]))
Node_class = list(range(max(data.y) + 2))
for run in range(runs):
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
    sel_basis = [[int(list(df[i].to_numpy()).count(1) >= int(len(df[i].index) * 0.1))
                  for i in range(len(df.columns))]
                 for df in data_by_class.values()]

    datasett=load_data(dataset_name)
    data = datasett[0]

    feature_names = [ii for ii in range(fe_len)]
    idx_train = [data.train_mask[i][run] for i in range(len(data.y))]
    train_index = np.where(idx_train)[0]
    idx_val = [data.val_mask[i][run] for i in range(len(data.y))]
    valid_index = np.where(idx_val)[0]
    idx_test = [data.test_mask[i][run] for i in range(len(data.y))]
    test_index = np.where(idx_test)[0]
    num_class = np.max(label)
    for idx_test in test_index:
        label[idx_test] = max(data.y) + 1

    Train = np.concatenate((train_index, valid_index))
    #print('Run= ',run)
    F_vec = spatial_embeddings(Node_class, Edgelist, Number_nodes,label)

    Fec, SFec = Contextual_embeddings(Data, basis, sel_basis, feature_names)
    concatenated_list = np.concatenate((Fec, SFec, F_vec,), axis=1)
    if dataset_name=='texas':
        acc_CC = ClassContrastTexas(concatenated_list, data.y, train_index, test_index, Feature_ratio,run)
    else:
        acc_CC = ClassContrast(concatenated_list, data.y, train_index, test_index, Feature_ratio)
    print(f'Run: {run + 1:02d},' f'Test Accuracy: {acc_CC:.2f}', "%\n")
    Accuracy_CC.append(acc_CC)

print(f'All runs:')
print(f'   Final Test: {statistics.mean(Accuracy_CC):.2f} ± {statistics.stdev(Accuracy_CC):.2f}')