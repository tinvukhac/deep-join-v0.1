import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_datasets_feature(filename):
    features_df = pd.read_csv(filename, delimiter=',', header=0)
    return features_df


def load_join_data(features_df, result_file):
    cols = ['dataset1', 'dataset2', 'result_size']
    result_df = pd.read_csv(result_file, delimiter=',', header=None, names=cols)
    result_df = pd.merge(result_df, features_df, left_on='dataset1', right_on='dataset_name')
    result_df = pd.merge(result_df, features_df, left_on='dataset2', right_on='dataset_name')
    result_df = result_df.drop(columns=['dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y'])

    x = result_df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    result_df = pd.DataFrame(x_scaled)

    return result_df


def main():
    print('Dataset utils')
    features_df = load_datasets_feature('data/datasets_features.csv')
    load_join_data(features_df, 'data/result_size.csv')


if __name__ == '__main__':
    main()
