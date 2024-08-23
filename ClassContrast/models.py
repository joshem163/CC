import numpy as np
from xgboost import XGBClassifier


def spatial_embeddings(Node_class, Edge_indices, n,label):
    F_vec = []
    for i in range(n):
        # print("\rProcessing file {} ({}%)".format(i, 100*i//(n-1)), end='', flush=True)
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


def Contextual_embeddings(DataFram, basis, sel_basis, feature_names):
    Fec = []
    SFec = []

    for i in range(len(DataFram)):
        vec = []
        Svec = []

        # Extract the features for the current node
        f = DataFram.loc[i, feature_names].values.flatten().tolist()

        # Compute similarities for basis
        for b in basis:
            vec.append(Similarity(f, b))

        # Compute similarities for sel_basis
        for sb in sel_basis:
            Svec.append(Similarity(f, sb))

        # Clear the feature list and append results
        f.clear()
        Fec.append(vec)
        SFec.append(Svec)

    return Fec, SFec

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def ContextualPubmed(DataAttribute):
    # Scale data before applying PCA
    scaling = StandardScaler()

    # Use fit and transform method
    scaling.fit(DataAttribute)
    Scaled_data = scaling.transform(DataAttribute)

    # Set the n_components=3
    m = 100
    principal = PCA(n_components=m)
    principal.fit(Scaled_data)
    x = principal.transform(Scaled_data)
    return x

def ClassContrast(attributes, labels, train_indices, test_indices,fr):
    feature = []
    for i in range(len(attributes[0])):
        feature.append("{}".format(i))
    #print(len(attributes[0]))
    X = attributes[:, :len(feature)]  # Features
    y = labels  # Labels

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    # Fit the model
    num_features_to_select = int(len(attributes[0]) * fr)
    model.fit(X_train, y_train)
    weight = model.get_booster().get_score(importance_type='weight')
    sorted_dict = {k: v for k, v in sorted(weight.items(), key=lambda item: (-item[1], item[0]))}
    best_features = list(sorted_dict.keys())[:num_features_to_select]
    #print(best_features)

    # Train using the best features
    X = attributes[:, :len(best_features)]  # Features based on selected best features
    X_train = X[train_indices]
    X_test = X[test_indices]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Don't cheat - fit only on training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # Apply the same transformation to test data
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(700,), random_state=1, max_iter=1000,
                        warm_start=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    from sklearn import metrics
    # Model Accuracy
    # print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100, "%\n")
    return metrics.accuracy_score(y_test, y_pred) * 100


def ClassContrastTexas(attributes, labels, train_indices, test_indices, fr,run):
    feature = []
    for i in range(len(attributes[0])):
        feature.append("{}".format(i))

    single_node = [0,1,2,7]
    print(len(attributes[0]))

    if run in single_node:
        #best_features = ['f16', 'f13', 'f17', 'f15', 'f14', 'f19', 'f18', 'f21', 'f22']
        best_features=['f8', 'f2', 'f9', 'f5', 'f1', 'f4', 'f10', 'f18', 'f0', 'f16']
    else:
        X = attributes[:, :len(feature)]  # Features
        y = labels  # Labels

        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
        # Fit the model
        num_features_to_select = int(len(attributes[0]) * fr)
        model.fit(X_train, y_train)
        weight = model.get_booster().get_score(importance_type='weight')
        sorted_dict = {k: v for k, v in sorted(weight.items(), key=lambda item: (-item[1], item[0]))}
        best_features = list(sorted_dict.keys())[:num_features_to_select]

    #print(best_features)

    # Train using the best features
    X = attributes[:, :len(best_features)]  # Features based on selected best features
    y = labels
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Don't cheat - fit only on training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # Apply the same transformation to test data
    X_test = scaler.transform(X_test)

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(700,), random_state=1, max_iter=1000,
                        warm_start=True)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    from sklearn import metrics
    # Model Accuracy
    # print("Test Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100, "%\n")
    return metrics.accuracy_score(y_test, y_pred) * 100




