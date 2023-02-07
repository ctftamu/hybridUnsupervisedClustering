import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


# Read data
filename = "data/SLM_data.csv"
df = pd.read_csv(filename)
cluster_features = df[['Zmean', 'Zvar', 'Chi', 'Temp', 'rho', 'diff', 'visc', 'CO', 'OH']]
cluster_vars_features = df[['Zmean', 'Zvar', 'Chi', 'Temp', 'rho', 'diff', 'visc']]
cluster_mf_features = df[['Zmean', 'Zvar', 'Chi', 'CO', 'OH']]

# Normalize clustering features
for j in range(len(cluster_vars_features[0])):
    mean = cluster_vars_features[:,[j]].mean()
    stdv = cluster_vars_features[:,[j]].std()
    cluster_vars_features[:,[j]] -= mean
    cluster_vars_features[:,[j]] /= stdv

for j in range(len(cluster_mf_features[0])):
    mean = cluster_mf_features[:,[j]].mean()
    stdv = cluster_mf_features[:,[j]].std()
    cluster_mf_features[:,[j]] -= mean
    cluster_mf_features[:,[j]] /= stdv


def cluster(n_clusters, cluster_features, filename):
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

    # Export cluster information to csv
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


''' Example '''
# cluster(4, cluster_mf_features, "SLM-Gaussian-4-mf.csv")
# cluster(4, cluster_vars_features, "SLM-Gaussian-4-vars.csv")
