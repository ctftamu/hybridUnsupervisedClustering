import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import List


def load_data(data_file: str, features: List[str]):
    """
    Loads data from a csv file and takes features to use for clustering.
    The data should be arranged by column, where each column represents
    a different feature.

    For example:
        ['Zmean', 'Zvar', 'Chi', 'Temp', 'rho', 'diff', 'visc']
        ['Zmean', 'Zvar', 'Chi', 'CO', 'OH']

    Args:
        data_file: string with data file path
        features: List of features as strings as defined in csv (N_feat)

    Returns:
        cluster_features: Numpy array of normalized features
            of shape (N_data, N_feat)
        means: Numpy array of feature means of shape (N_feat,)
        stdvs: Numpy array of feature standard deviations of
            shape (N_feat,)
    """
    # Read data
    df = pd.read_csv(data_file)
    cluster_features = df[features].to_numpy()

    # Normalize clustering features
    means = cluster_features.mean(axis=0)
    stdvs = cluster_features.std(axis=0)
    cluster_features -= means
    cluster_features /= stdvs

    return cluster_features, means, stdvs


def cluster_data(n_clusters: int, cluster_features: np.ndarray, filename: str):
    """
    Clusters data using a Gaussian Mixture Model and stores clustering
    information in a csv.

    Example usage:
        cluster_data(4, cluster_mf_features, "SLM-Gaussian-4-mf.csv")
        cluster_data(10, cluster_vars_features, "SLM-Gaussian-10-vars.csv")

    Args:
        n_clusters: int of number of clusters
        cluster_features: Numpy array of normalized features
            of shape (N_data, N_feat)
        filename: string for new file path

    Returns:
        None
    """
    # Fit GMM
    model = GaussianMixture(n_components=n_clusters)
    model.fit(cluster_features)

    # Predict clusters for each data point
    yhat = model.predict(cluster_features)
    clusters = np.unique(yhat)
    rows = []
    for cluster in clusters:
        # get row indexes for samples with this cluster
        rows.append(np.where(yhat == cluster))

    # Separate data into clusters based on predictions
    features_clusters = []
    for cluster in range(len(clusters)):
        for i in range(len(rows[cluster][0])):
            feature_n = cluster_features[rows[cluster][0][i]].copy()
            feature_n = np.append(feature_n, [(cluster+1)])
            features_clusters.append(feature_n)

    # Export cluster information to csv in same order as imported
    # For variables:      cluster_num, Zmean, Zvar, Chi, Temp, rho, diff, visc
    # For mass fractions: cluster_num, Zmean, Zvar, Chi, CO, OH
    newfile = open(filename, "w+")
    newfile.write(str(len(clusters))+"\n")
    for i in range(len(features_clusters)):
        newfile.write(str(int(features_clusters[i][-1])) + ",")
        for j in range(len(features_clusters[0])):
            if (j == len(features_clusters[0])-1):
                newfile.write(str(features_clusters[i][j]) + "\n")
            else:
                newfile.write(str(features_clusters[i][j]) + ",")

    newfile.close()
