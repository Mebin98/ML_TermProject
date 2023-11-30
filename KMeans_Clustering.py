import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def load_dataset():
    """Load datasets from Google Drive."""
    data_std = pd.read_csv('/content/gdrive/My Drive/ML_PHW1/Project/K-means/scaled_data_standard.csv')
    data_minmax = pd.read_csv('/content/gdrive/My Drive/ML_PHW1/Project/K-means/scaled_data_minmax.csv')
    data_normalizer = pd.read_csv('/content/gdrive/My Drive/ML_PHW1/Project/K-means/scaled_data_normalizer.csv')
    data_quantile = pd.read_csv('/content/gdrive/My Drive/ML_PHW1/Project/K-means/scaled_data_quantile.csv')
    data_robust = pd.read_csv('/content/gdrive/My Drive/ML_PHW1/Project/K-means/scaled_data_robust.csv')
    return data_std, data_minmax, data_normalizer, data_quantile, data_robust

def plot_elbow_method(data):
    """Plot the elbow method graph for determining optimal number of clusters."""
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def cluster_data_then_plot(original_data):
    """Cluster data and plot the results."""
    data = original_data.drop('id', axis=1)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=0)
        clusters = kmeans.fit_predict(data)
        data_pca_with_clusters = np.column_stack((data_pca, clusters))
        plt.figure(figsize=(8, 6))
        for cluster in np.unique(clusters):
            row_ix = np.where(clusters == cluster)
            plt.scatter(data_pca_with_clusters[row_ix, 0], data_pca_with_clusters[row_ix, 1])
        centroids = pca.transform(kmeans.cluster_centers_)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', label='Centroids')
        plt.title(f'K-means Clustering with PCA (k={k})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()

def plot_silhouette(X, n_clusters, cluster_labels):
    """Plot silhouette scores for each cluster."""
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

def cluster_data_then_evaluate(original_data):
    """Cluster data and evaluate the results using silhouette scores."""
    data = original_data.drop('id', axis=1)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    plot_elbow_method(data)
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=0)
        clusters = kmeans.fit_predict(data)
        plot_silhouette(data, k, clusters)

def main():
    """Main function to load data, cluster, and evaluate results."""
    std, minmax, normal, quantile, robust = load_dataset()
    datasets = [std, minmax, normal, quantile, robust]
    for dataset in datasets:
        cluster_data_then_evaluate(dataset)

if __name__ == "__main__":
    main()
