import datasets

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing


def main():
    print ('Linear regression model to predict join time')

    num_rows, num_columns = 16, 16
    datasets_features_path = 'data/spatial_descriptors/spatial_descriptors_small_datasets.csv'
    datasets_histograms_path = 'data/histograms/small_datasets'
    join_results_path = 'data/join_results/join_results_small_datasets_no_bit.csv'
    features_df = datasets.load_datasets_feature(datasets_features_path)
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data(
        features_df,
        join_results_path,
        datasets_histograms_path,
        num_rows, num_columns)

    print (join_data)

    join_data = join_data[['cardinality_x', 'cardinality_y', 'result_size', 'mbr_tests', 'duration']]
    duration = join_data['duration']
    join_data = join_data.drop(columns=['duration'])
    x = join_data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    join_data = pd.DataFrame(x_scaled)
    join_data.insert(len(join_data.columns), 'duration', duration, True)
    print (join_data)

    train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_all_histogram_train, ds_all_histogram_test = train_test_split(
        join_data, ds1_histograms, ds2_histograms, ds_all_histogram, test_size=0.20, random_state=42)

    X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(2, 4)]])
    X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(2, 4)]])
    y_train = train_attributes['duration']
    y_test = test_attributes['duration']

    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ('mean = {}, std = {}'.format(mean, std))


if __name__ == '__main__':
    main()
