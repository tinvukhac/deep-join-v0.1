import datasets

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from itertools import combinations


def main():
    print ('Train and test join data using baseline model')

    num_rows, num_columns = 16, 16
    # target = 'join_selectivity'
    target = 'mbr_tests_selectivity'
    datasets_features_path = 'data/spatial_descriptors/spatial_descriptors_small_datasets.csv'
    datasets_histograms_path = 'data/histograms/small_datasets'
    join_results_path = 'data/join_results/join_results_small_datasets_no_bit.csv'
    features_df = datasets.load_datasets_feature(datasets_features_path)
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data(
        features_df,
        join_results_path,
        datasets_histograms_path,
        num_rows, num_columns)

    train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_all_histogram_train, ds_all_histogram_test = train_test_split(
        join_data, ds1_histograms, ds2_histograms, ds_all_histogram, test_size=0.20, random_state=42)

    num_features = len(train_attributes.columns) - 10
    # print (join_data)
    X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(num_features)]])
    X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(num_features)]])
    y_train = train_attributes[target]
    y_test = test_attributes[target]

    print ('Test on small datasets')
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ('mean = {}, std = {}'.format(mean, std))

    print ('Test on large datasets')
    datasets_features_path = 'data/spatial_descriptors/spatial_descriptors_large_datasets.csv'
    datasets_histograms_path = 'data/histograms/large_datasets'
    join_results_path = 'data/join_results/join_results_large_datasets_no_bit.csv'
    features_df = datasets.load_datasets_feature(datasets_features_path)
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data(
        features_df,
        join_results_path,
        datasets_histograms_path,
        num_rows, num_columns)

    X_test = pd.DataFrame.to_numpy(join_data[[i for i in range(num_features)]])
    y_test = join_data[target]
    y_pred = reg.predict(X_test)

    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ('mean = {}, std = {}'.format(mean, std))

    # features_df = datasets.load_datasets_feature('data/uniform_datasets_features.csv')
    # join_data, ds1_histograms, ds2_histograms, ds_all_histogram = datasets.load_join_data(features_df,
    #                                                                                       'data/uniform_result_size.csv',
    #                                                                                       'data/histogram_uniform_values',
    #                                                                                       16, 16)

    # output_f = open('model_performance.csv', 'w')
    # # distributions = [['diagonal', 'uniform'], ['gaussian', 'uniform'], ['parcel', 'uniform']]
    # distributions = ['diagonal', 'gaussian', 'parcel', 'uniform']
    # # distributions = [['diagonal', 'uniform'], ['gaussian', 'uniform'], ['parcel', 'uniform']]
    #
    # for r in range(1, len(distributions) + 1):
    #     groups = combinations(distributions, r)
    #     for dists in groups:
    #         features_df = datasets.load_datasets_feature('data/datasets_features.csv')
    #         # join_data, ds1_histograms, ds2_histograms, ds_all_histogram = datasets.load_join_data(features_df,
    #         #                                                                                       'data/result_size_{}_{}_{}.csv'.format(dist[0], dist[1], dist[2]),
    #         #                                                                                       'data/histogram_values',
    #         #                                                                                       16, 16)
    #         join_data, ds1_histograms, ds2_histograms, ds_all_histogram = datasets.load_join_data(features_df,
    #                                                                                               'data/result_size_{}.csv'.format('_'.join(dists)),
    #                                                                                               'data/histogram_values',
    #                                                                                               16, 16)
    #
    #         # join_data.drop(['dataset1', 'dataset2'])
    #
    #         train_attributes, test_attributes = train_test_split(join_data, test_size=0.25, random_state=42)
    #
    #         # Exclude BOPS and result_size
    #         num_features = len(join_data.columns) - 7
    #         X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(num_features)]])
    #         X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(num_features)]])
    #         # y_train = train_attributes['result_size']
    #         # y_test = test_attributes['result_size']
    #         y_train = train_attributes['join_selectivity']
    #         y_test = test_attributes['join_selectivity']
    #
    #         reg = LinearRegression().fit(X_train, y_train)
    #         y_pred = reg.predict(X_test)
    #         print (dists)
    #         print ('r2 score: {}'.format(r2_score(y_test, y_pred)))
    #
    #         diff = y_pred.flatten() - y_test
    #         percent_diff = (diff / y_test)
    #         abs_percent_diff = np.abs(percent_diff)
    #         mean = np.mean(abs_percent_diff)
    #         std = np.std(abs_percent_diff)
    #
    #         print ('mean = {}, std = {}'.format(mean, std))
    #
    #         print ('{}\t{}\t{}\t{}'.format(dists, r2_score(y_test, y_pred), mean, std))
    #         output_f.writelines('{}\t{}\t{}\t{}\n'.format(dists, r2_score(y_test, y_pred), mean, std))
    #
    # output_f.close()

    # y_test_array = np.asarray(y_test)
    # y_pred_array = np.asarray(y_pred)
    #
    # y_test_array = y_test_array.reshape((len(y_test_array), 1))
    # y_pred_array = y_pred_array.reshape((len(y_pred_array), 1))
    #
    # np.savetxt('baseline_prediction_all_uniform.csv', np.concatenate([y_test_array, y_pred_array], axis=1), fmt='%.3f', delimiter=',')
    #
    # features_df_all = datasets.load_datasets_feature('data/datasets_features.csv')
    # join_data_all, ds1_histograms, ds2_histograms, ds_all_histogram = datasets.load_join_data(features_df_all,
    #                                                                                           'data/result_size.csv',
    #                                                                                           'data/histogram_values',
    #                                                                                           16, 16)
    # # join_data_all.drop(['dataset1', 'dataset2'])
    # train_attributes_all, test_attributes_all = train_test_split(join_data_all, test_size=0.25, random_state=42)
    # X_train_all = pd.DataFrame.to_numpy(train_attributes_all[[i for i in range(num_features)]])
    # X_test_all = pd.DataFrame.to_numpy(test_attributes_all[[i for i in range(num_features)]])
    # # y_train = train_attributes['result_size']
    # # y_test = test_attributes['result_size']
    # y_train_all = train_attributes_all['join_selectivity']
    # y_test_all = test_attributes_all['join_selectivity']
    #
    # y_pred_all = reg.predict(X_test_all)
    #
    # test_attributes_all.insert(len(test_attributes_all.columns), 'predicted_join_selectivity', y_pred_all, True)
    #
    # bops_all = test_attributes_all['bops']
    # result_size_all = y_pred_all * bops_all
    # test_attributes_all.insert(len(test_attributes_all.columns), 'predicted_size', result_size_all, True)
    # # test_attributes_all['predicted_size'] = result_size_all
    # # print (result_size_all)
    # test_attributes_all.to_csv('test_attributes_all.csv')

    # join_data_dia_gaus, ds1_histograms, ds2_histograms, ds_all_histogram = datasets.load_join_data(features_df_all,
    #                                                                                       'data/result_size_dia_gaus.csv',
    #                                                                                       'data/histogram_values',
    #                                                                                       16, 16)
    # X_test_dia_gaus = pd.DataFrame.to_numpy(join_data_dia_gaus[[i for i in range(num_features)]])
    # y_pred_dia_gaus = reg.predict(X_test_dia_gaus)
    # print (y_pred_dia_gaus)
    # np.savetxt('dia_gaus_prediction.csv', y_pred_dia_gaus, delimiter=',')


if __name__ == '__main__':
    main()
