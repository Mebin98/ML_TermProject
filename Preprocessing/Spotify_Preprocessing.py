import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler, Normalizer

# load dataset
# Depend of Directory: this code would be different!
def load_dataset():
    original_data = pd.read_csv('data.csv')
    return original_data

# by applying spotify open api
# We merged music data from 2021-2023.
def merge_crt_data(data):
    crt_data = pd.read_csv('crt_data.csv')
    merged_data = pd.concat([data,crt_data])
    return merged_data

# convert ms to minute unit
def ms_to_minutes(data):
    # milliseconds -> seconds
    data['duration_ms'] = data['duration_ms'] / 1000
    # seconds -> minute
    data['duration_ms'] = data['duration_ms'] / 60
    return data

# explicit : Data Imbalance, binary, and Not considered an important feature for music recommendation
# release_data : redudant with year feature
# name : It is not used when clustering, However, keep it in original_data
# Well, artist is important feature, but,,, there are a lots of singers though
# So, delete artist feature
delete_features = ['name','explicit','release_date','artists']
def delete_unusable_features(data,features):
    for feature in features:
        if feature in data.columns:
            data = data.drop(feature, axis=1)
    return data

# We judged that old songs would not be selected to customers
# So under 1960, delete
# In this function, it also deletes the rows that has popularity value as '0'
def delete_old_songs(data):
    # Keep only the rows where the 'year' is 1960 or later
    data = data[data['year'] >= 1960]
    data = data[data['popularity'] != 0]
    return data

# outlier > 10,000 -> delete feature
# else, delete outliers
def handle_outlier(data):
    # Select numeric columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

    # Set plotting parameters
    nrows = 5
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 20))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    # Lists to track columns and rows for deletion
    columns_to_delete = []
    rows_to_delete = set()

    for i, column in enumerate(numerical_columns):
        # Plot boxplot
        row = i // ncols
        col = i % ncols
        data.boxplot(column=column, ax=axes[row, col])
        axes[row, col].set_title(column)

        # Calculate outliers using IQR
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
        outlier_indices = data[outlier_condition].index

        # Count outliers and determine action
        outlier_count = len(outlier_indices)
        if outlier_count > 10000:
            columns_to_delete.append(column)
        else:
            rows_to_delete.update(outlier_indices)

        print(f"{column} number of outliers: {outlier_count}")

    # Hide empty subplot areas
    for j in range(len(numerical_columns), nrows*ncols):
        fig.delaxes(axes[j // ncols, j % ncols])

    plt.show()

    # Remove identified columns and rows from the dataset
    data.drop(columns=columns_to_delete, inplace=True)
    data.drop(index=rows_to_delete, inplace=True)

    return data



def count_outliers(data):
    # Select numeric columns
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    outlier_counts = {}

    for column in numerical_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        outlier_condition = (data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR))
        outlier_count = len(data[outlier_condition])
        outlier_counts[column] = outlier_count
        
        # Print the number of outliers for each column
        print(f"{column} : {outlier_count} outliers")



def plot_outliers(data):
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    nrows = (len(numerical_columns) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(15, nrows * 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, column in enumerate(numerical_columns):
        row = i // 3
        col = i % 3
        data.boxplot(column=column, ax=axes[row, col])
        axes[row, col].set_title(column)

    # Hide empty subplot areas if they exist
    for j in range(len(numerical_columns), nrows * 3):
        fig.delaxes(axes[j // 3, j % 3])
    plt.show()

# decided to delete liveness feature
# a lot of outliers, and not important
# also deletes mode feature that has only 1 and 0 values
def delete_liveness_mode(data):
    data = data.drop('liveness',axis=1)
    data = data.drop('mode',axis=1)
    return data


def scale_data_to_csv(data, file_prefix):
    # Separate the 'id' column and reset its index
    id_column = data['id'].reset_index(drop=True)
    data_without_id = data.drop('id', axis=1)

    # Define scalers
    scalers = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
        'quantile': QuantileTransformer(),
        'robust': RobustScaler(),
        'normalizer': Normalizer()
    }

    # Apply each scaler and save the result as a CSV file with the 'id' column added back
    for name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(data_without_id)
        scaled_df = pd.DataFrame(scaled_data, columns=data_without_id.columns)
        scaled_df.insert(0, 'id', id_column)  # Add the 'id' column back
        scaled_df.to_csv(f'{file_prefix}_{name}.csv', index=False)




if __name__ == '__main__':
    original_data = load_dataset() # load dataset
    data = merge_crt_data(original_data) # merge data from current data(2021-2023)
    data = ms_to_minutes(data)
    data = delete_unusable_features(original_data, delete_features) # delete unusable features for clustering
    data = delete_old_songs(data) # delete old songs
    data = handle_outlier(data) # through this step, duration_ms and instrumentalness are deleted
    count_outliers(data) # checks outliers again
    plot_outliers(data) # plot outliers again
    data = delete_liveness_mode(data) # cause of 3000 outliers, delete liveness feature, and also deletes mode
    scale_data_to_csv(data, 'scaled_data') # scale data and save to csv

