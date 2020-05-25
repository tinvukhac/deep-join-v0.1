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

    # Load histograms
    ds1_histograms, ds2_histograms, ds1_original_histograms, ds2_original_histograms = load_histograms(result_df, 16, 16)

    # Compute BOPS
    bops = np.multiply(ds1_original_histograms, ds2_original_histograms)
    bops = bops.reshape((bops.shape[0], 256))
    bops_values = np.sum(bops, axis=1)
    bops_values = bops_values.reshape((bops_values.shape[0], 1))
    result_df['bops'] = bops_values
    # print (bops_values)
    result_size = result_df['result_size']
    result_df = result_df.drop(columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y'])

    # x = result_df.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # result_df = pd.DataFrame(x_scaled)

    result_df.insert(19, 'result_size', result_size, True)

    print (result_df)
    result_df.to_csv('result_df.csv')

    return result_df, ds1_histograms, ds2_histograms


def load_histogram(num_rows, num_columns, dataset):
    hist = np.genfromtxt('data/histogram_values/{}x{}/{}'.format(num_rows, num_columns, dataset), delimiter=',')
    normalized_hist = hist / hist.max()
    normalized_hist = normalized_hist.reshape((hist.shape[0], hist.shape[1], 1))
    hist = hist.reshape((hist.shape[0], hist.shape[1], 1))
    return normalized_hist, hist


def load_histograms(result_df, num_rows, num_columns):
    ds1_histograms = []
    ds2_histograms = []
    ds1_original_histograms = []
    ds2_original_histograms = []

    for dataset in result_df['dataset1']:
        normalized_hist, hist = load_histogram(num_rows, num_columns, dataset)
        ds1_histograms.append(normalized_hist)
        ds1_original_histograms.append(hist)

    for dataset in result_df['dataset2']:
        normalized_hist, hist = load_histogram(num_rows, num_columns, dataset)
        ds2_histograms.append(normalized_hist)
        ds2_original_histograms.append(hist)

    return np.array(ds1_histograms), np.array(ds2_histograms), np.array(ds1_original_histograms), np.array(ds2_original_histograms)


def main():
    print('Dataset utils')
    features_df = load_datasets_feature('data/datasets_features.csv')
    load_join_data(features_df, 'data/result_size.csv')


if __name__ == '__main__':
    main()
