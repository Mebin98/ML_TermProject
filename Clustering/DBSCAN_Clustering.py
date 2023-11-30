import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

# Function to visualize silhouette scores for different DBSCAN clustering settings
def visualize_silhouette(dbscan_arg_list, X_features):
    """
    Visualizes silhouette scores for different DBSCAN clustering configurations.
    Args:
    dbscan_arg_list: List of tuples (eps, min_samples) for DBSCAN.
    X_features: Feature set for clustering.
    """
    n_cols = len(dbscan_arg_list)
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    cluster_objs = []

    for ind, (eps, min_samples) in enumerate(dbscan_arg_list):
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
        cluster_labels = dbscan.fit_predict(X_features)
        cluster_objs.append(dbscan)

        # Silhouette scores
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        n_cluster = len(np.unique(cluster_labels))

        # Plot settings
        y_lower = 10
        axs[ind].set_title(f'Number of Cluster: {n_cluster}\nSilhouette Score: {round(sil_avg,3)}\neps: {eps}, min_points: {min_samples}')
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # Plot silhouette scores for each cluster
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels == i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

    plt.show()
    return cluster_objs

# Function to check correlations in the feature set
def check_corr(X_features):
    """
    Displays a heatmap of correlations between features in the dataset.
    Args:
    X_features: DataFrame containing the features.
    """
    import seaborn as sns
    corr = X_features.corr()
    plt.figure(figsize=(14,14))
    sns.heatmap(corr, annot=True, fmt='.1g')

# Function to load different datasets
def load_dataset():
    """
    Loads multiple datasets with different scaling techniques.
    Returns:
    Tuple of DataFrames for each scaled dataset.
    """
    import pandas as pd
    data_std = pd.read_csv('dataset_standard.csv')
    data_minmax = pd.read_csv('dataset_minmax.csv')
    data_normalizer = pd.read_csv('dataset_normalizer.csv')
    data_quantile = pd.read_csv('dataset_quantile.csv')
    data_robust = pd.read_csv('dataset_robust.csv')
    return data_std, data_minmax, data_normalizer, data_quantile, data_robust

# Function to visualize clustering results
def visualize_cluster_plot(clusterobj, dataframe, iscenter=True):
    """
    Visualizes clustering results using DBSCAN.
    Args:
    clusterobj: DBSCAN clustering object.
    dataframe: DataFrame with clustering results.
    iscenter: Flag to show cluster centers.
    """
    centers = clusterobj.labels_
    label_name = 'center'
    dataframe[label_name] = centers

    unique_labels = np.unique(dataframe[label_name].values)
    markers=['o', 's', '^', 'x', '*']
    isNoise=False

    for label in unique_labels:
        label_cluster = dataframe[dataframe[label_name]==label]
        if label == -1:
            cluster_legend = 'Noise'
            isNoise=True
        else:
            cluster_legend = 'Cluster '+str(label)
        
        plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], s=70, edgecolor='k', marker=markers[label], label=cluster_legend)
        
        if iscenter:
            center_x_y = centers[label]
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=250, color='white', alpha=0.9, edgecolor='k', marker=markers[label])
            plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='k', edgecolor='k', marker='$%d$' % label)
    
    legend_loc = 'upper center' if isNoise else 'upper right'
    plt.legend(loc=legend_loc)
    plt.show()

# PCA transformation function
def pca_transform(data, columns, n_axes=2):
    """
    Performs PCA transformation on the dataset.
    Args:
    data: DataFrame to be transformed.
    columns: Columns to be included in the PCA.
    n_axes: Number of principal components.
    Returns:
    Transformed DataFrame.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_axes)
    pca_data = pca.fit_transform(data[columns])
    print('PCA Component별 변동성:', pca.explained_variance_ratio_)

    pca_features = ['ftr' + str(i) for i in range(1, 1 + n_axes)]
    pca_df = pd.DataFrame(pca_data, columns=pca_features)
    return pca_df

# Main execution flow
data_std, data_minmax, data_normalizer, data_quantile, data_robust = load_dataset()
selected_columns = ['valence', 'acousticness', 'danceability', 'energy', 'popularity', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo']
dbscan_args = [(eps, min_samples) for eps in np.arange(0.03, 1, step=0.25) for min_samples in range(3, 10, 1)]
visualize_silhouette(dbscan_arg_list=dbscan_args, X_features=data_std[selected_columns])

pca_dfs = [pca_transform(data, selected_columns, n_axes=2) for data in [data_std, data_minmax, data_normalizer, data_quantile, data_robust]]
cluster_objs, pca_df, feature_names = visualize_silhouette(dbscan_arg_list=dbscan_args, X_features=pca_dfs[4])
for cluster_obj in cluster_objs:
    visualize_cluster_plot(clusterobj=cluster_obj, dataframe=pca_df)
