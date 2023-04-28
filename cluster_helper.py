import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools

def plot_silhoutte_score(X, k_max=20):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    num_clusters = range(2, k_max + 1)
    sil_score = []
    for n in num_clusters:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X)
        preds = kmeans.predict(X)
        sil_score.append(silhouette_score(X, preds))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(num_clusters, sil_score)
    plt.title("Silhoutte Score")
    plt.xlabel("Number of Clusters")
    plt.xticks(range(1, k_max + 1))
    plt.show()

def under_partition_measure(X, k_max):
    from sklearn.cluster import KMeans
    ks = range(1,k_max+1)
    UPM = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        UPM.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(ks, UPM);
    plt.show()
    return UPM

def over_partition_measure(X, k_max):
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances
    ks = range(1, k_max + 1)
    OPM = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        d_min = np.inf
        for pair in list(itertools.combinations(centers, 2)):
            d = pairwise_distances(pair[0].reshape(1, -1), pair[1].reshape(1, -1), metric='euclidean')
            if d < d_min:
                d_min = d
        OPM.append(k / d_min)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(ks, OPM)
    plt.show()
    return OPM

def validity_index(X, k_max):
    UPM = under_partition_measure(X, k_max)
    OPM = over_partition_measure(X, k_max)
    UPM_min = np.min(UPM)
    OPM_min = np.min(OPM)
    UPM_max = np.max(UPM)
    OPM_max = np.max(OPM)
    norm_UPM = []
    norm_OPM = []
    for i in range(k_max):
        norm_UPM.append((UPM[i] - UPM_min) / (UPM_max - UPM_min))
        norm_OPM.append((OPM[i] - OPM_min) / (OPM_max - OPM_min))

    val_idx = np.array(norm_UPM).reshape(-1) + np.array(norm_OPM).reshape(-1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(range(1, k_max + 1), val_idx)
    plt.title("Validity Index")
    plt.xlabel("Number of Clusters")
    plt.xticks(range(1, k_max + 1))

    return val_idx