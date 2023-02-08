# hybridUnsupervisedClustering
Code for creating hybrid unsupervised clustering ML architecture for flamelet combustion models.


![Alt text](https://github.com/ctftamu/hybridUnsupervisedClustering/raw/main/images/Detailed_MLArchitecture.png)
![Alt text](https://github.com/ctftamu/hybridUnsupervisedClustering/raw/main/images/flame1.png)


## How to use the code
### Clustering
The file `clustering-SLM.py` holds the functions necessary to cluster flamelet data. 
First, the flamelet data must be properly formatted in a `.csv` file as follows:

| Feature 1 | ... | Feature n |
| :---: | :---: | :---: |
| $feature_1^1$ | ... | $feature_1^n$ |
| $\vdots$ | ... | $\vdots$ |
| $feature_N^1$ | ... | $feature_N^n$ |

with N total data points and n total features.

Then, using the function `load_data`, the data will be normalized and stored in an array with shape `(N,n)`. Next, the features are clustered based on a user-specified number of clusters using the function `cluster_data`. The clustering information is exported as a `.csv` file. An example of this clustering flow would be:

```python
# Clustering for various simulation variables
cluster_vars_features, var_means, var_stdvs = load_data(data_file="SLM_data.csv", features=['Zmean', 'Zvar', 'Chi', 'Temp', 'rho', 'diff', 'visc'])
cluster_data(n_clusters=10, cluster_features=cluster_vars_features, filename="SLM-10-vars.csv")

# Clustering for mass fractions
cluster_mf_features, mf_means, mf_stdvs = load_data(data_file="SLM_data.csv", features=['Zmean', 'Zvar', 'Chi', 'CO', 'OH'])
cluster_data(n_clusters=4, cluster_features=cluster_mf_features, filename="SLM-4-mf.csv")
```

### Training ML models
The file `NN-training.py` holds the functions for training the expert models and gating network to implement the clusterwise regression model.

The clustered data exported by the function `cluster_data` is loaded in by the function `read_data`. The functions `train_experts` and `train_gating_net` train the appropriate NN models using the Keras framework. Finally, the `evaluate_models` function can be used to test the gating network and expert models in conjuction to observe how the clusterwise regression model performs holistically.

These training functions can be used as follows:
```python
# Read in the data from cluster data file
train_features, test_features, train_targets, test_targets, train_clusters, test_clusters = read_data(data_file="SLM-10-vars.csv", n_inputs=3, n_outputs=4)

# Further data organization
combined_train_features, combined_test_features, combined_train_targets, combined_test_targets, combined_train_clusters, combined_test_clusters = combine_cluster_data(train_features, test_features, train_targets, test_targets, train_clusters, test_clusters)

# Train expert models
expert_models = train_experts(train_features, test_features, train_targets, test_targets)

# Train gating network
gating_net = train_gating_net(combined_train_features, combined_test_features, combined_train_clusters, combined_test_clusters)

# Check model performance
evaluate_model(expert_models, gating_net, combined_test_features, combined_test_targets, combined_test_clusters, var_stdvs)
```

These NN models, saved as `.h5` files, can be converted to a `.txt` format to be used with the Fortran-Keras bridge and integrated into Fortran-based software.

