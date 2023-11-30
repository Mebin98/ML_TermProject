import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
#from google.colab import drive
#drive.mount('/content/gdrive')

def load_dataset():
    # this could be different based on directory!
    data_std = pd.read_csv('scaled_data_standard.csv')
    data_minmax = pd.read_csv('scaled_data_minmax.csv')
    data_normalizer = pd.read_csv('scaled_data_normalizer.csv')
    data_quantile = pd.read_csv('scaled_data_quantile.csv')
    data_robust = pd.read_csv('scaled_data_robust.csv')
    return data_std, data_minmax, data_normalizer, data_quantile, data_robust

def plot_elbow_method(original_data):
    """Plot the elbow method graph for determining optimal number of clusters."""
    # Remove 'id' column if it exists
    if 'id' in original_data.columns:
        data = original_data.drop('id', axis=1)
    else:
        data = original_data

    distortions = []
    for k in range(2, 6):
        spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')
        labels = spectral.fit_predict(data)
        # Compute distortion
        distortions.append(sum(np.min(cdist(data, spectral.fit_transform(data), 'euclidean'), axis=1)) / data.shape[0])

    plt.plot(range(2, 6), distortions)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()

def plot_clusters(data_pca, labels):
    """Plots the PCA reduced data with clusters."""
    unique_clusters = np.unique(labels)
    plt.figure(figsize=(8, 6))
    for cluster in unique_clusters:
        plt.scatter(data_pca[labels == cluster, 0], data_pca[labels == cluster, 1], label=f'Cluster {cluster}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clusters in PCA-reduced Space')
    plt.legend()
    plt.show()

def plot_silhouette(data, cluster_labels):
    """Plot silhouette scores for each cluster."""
    silhouette_avg = silhouette_score(data, cluster_labels)
    print(f"The average silhouette_score is : {silhouette_avg}")
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    y_lower = 10
    for i in np.unique(cluster_labels):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / len(np.unique(cluster_labels)))
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 10
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

def cluster_data_then_evaluate_with_spectral(original_data):
    """Cluster data using Spectral Clustering and evaluate the results."""
    # Remove 'id' column if it exists
    if 'id' in original_data.columns:
        data_without_id = original_data.drop('id', axis=1)
    else:
        data_without_id = original_data

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_without_id)

    for k in range(2, 6):
        spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')
        labels = spectral.fit_predict(data_pca)

        # Plot clusters
        plot_clusters(data_pca, labels)

        # Silhouette score
        plot_silhouette(data_pca, labels)

def simulate_all_data():
    std, minmax, normal ,quantile,robust = load_dataset()

    datasets = [std,minmax,normal,quantile,robust]
    for dataset in datasets:
        # Clustering and evaluation
        #cluster_data_then_evaluate_with_spectral(dataset)

        # Elbow method
        plot_elbow_method(dataset)

simulate_all_data()