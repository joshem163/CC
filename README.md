# ClassContrast
![framework](https://github.com/joshem163/ClassContrast/assets/133717791/c50235ad-e30c-4b87-86d0-2c3cc1e37d3c)


# Overview

ClassContrast is an innovative machine learning model for Graph representation learning that utilizes a contrastive approach, effectively integrating local neighborhood information and domain features of the nodes. This repository contains the code for the ClassContrast paper.
# Requirements
ClassContrast depends on the followings:
1. Pytorch
2. Pyflagser
3. networkx 3.1
4. ogb 1.3.6
5. sklearn 1.3.0
6. torch_geometric 2.4.0
7. natsort 8.4.0
   
The code is implemented in python 3.11.4. 
# Datasets
In this study,  ten benchmark datasets have been utilized, comprising three homophilic, two OGBN, and five heterophilic datasets. The link to access these datasets is provided below:

[CORA](https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.zip) 

[CITESEER](https://linqs-data.soe.ucsc.edu/public/datasets/citeseer-doc-classification/citeseer-doc-classification.zip)

[OGBN](https://ogb.stanford.edu/docs/nodeprop/)

[PUBMED](https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.zip)

[WebKB](https://github.com/bingzhewei/geom-gcn/tree/master/new_data)

[Wiki](https://github.com/benedekrozemberczki/MUSAE/tree/master/input)

# Runing the  Experiments
To repeat the experiment, run the train.py file with the corresponding dataset and other hyperparameter.  


