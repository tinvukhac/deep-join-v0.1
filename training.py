from keras import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import concatenate
import keras

import datasets
import models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math


def main():
    print ('Training the join cardinality estimator')
    is_train = True
    num_rows, num_columns = 16, 16
    # target = 'join_selectivity'
    target = 'mbr_tests_selectivity'
    datasets_features_path = 'data/spatial_descriptors/spatial_descriptors_small_datasets.csv'
    datasets_histograms_path = 'data/histograms/small_datasets'
    join_results_path = 'data/join_results/join_results_small_datasets_no_bit.csv'
    features_df = datasets.load_datasets_feature(datasets_features_path)
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data(features_df,
                                                                                          join_results_path,
                                                                                          datasets_histograms_path,
                                                                                          num_rows, num_columns)

    train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_all_histogram_train, ds_all_histogram_test, ds_bops_histogram_train, ds_bops_histogram_test = train_test_split(
        join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram, test_size=0.20, random_state=42)

    # train_attributes, val_attributes, ds1_histograms_train, ds1_histograms_val, ds2_histograms_train, ds2_histograms_val, ds_all_histogram_train, ds_all_histogram_val = train_test_split(
    #     train_attributes, ds1_histograms_train, ds2_histograms_train, ds_all_histogram_train, test_size=0.20, random_state=32)

    num_features = len(train_attributes.columns) - 10
    # print (join_data)
    X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(num_features)]])
    X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(num_features)]])
    y_train = train_attributes[target]
    y_test = test_attributes[target]
    # y_train = train_attributes['result_size']
    # y_test = test_attributes['result_size']

    mlp = models.create_mlp(X_train.shape[1], regress=False)
    cnn1 = models.create_cnn(num_rows, num_columns, 1, regress=False)
    # cnn2 = models.create_cnn(num_rows, num_columns, 1, regress=False)
    # cnn3 = models.create_cnn(num_rows, num_columns, 1, regress=False)

    # combined_input = concatenate([mlp.output, cnn1.output, cnn2.output, cnn3.output])
    combined_input = concatenate([mlp.output, cnn1.output])

    x = Dense(4, activation="relu")(combined_input)
    x = Dense(1, activation="linear")(x)

    # model = Model(inputs=[mlp.input, cnn1.input, cnn2.input, cnn3.input], outputs=x)
    model = Model(inputs=[mlp.input, cnn1.input], outputs=x)

    EPOCHS = 40
    LR = 1e-2
    # opt = Adam(lr=1e-4, decay=1e-4 / 200)
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # print (model.summary())

    # train the model
    if is_train:
        print("[INFO] training model...")
        # model.fit(
        #     [X_train, ds1_histograms_train, ds2_histograms_train], y_train,
        #     validation_data=([X_test, ds1_histograms_test, ds2_histograms_test], y_test),
        #     epochs=EPOCHS, batch_size=128)
        model.fit(
            [X_train, ds_bops_histogram_train], y_train,
            validation_data=([X_test, ds_bops_histogram_test], y_test),
            epochs=EPOCHS, batch_size=256)

        model.save('trained_models/model.h5')
        model.save_weights('trained_models/model_weights.h5')
    else:
        model = keras.models.load_model('trained_models/model.h5')
        model.load_weights('trained_models/model_weights.h5')

    print ('Test on small datasets')
    y_pred = model.predict([X_test, ds_bops_histogram_test])

    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    # test_attributes['join_selectivity_pred'] = y_pred
    # test_attributes['percent_diff'] = abs_percent_diff
    # test_attributes.to_csv('prediction_small.csv')

    # compute the mean and standard deviation of the absolute percentage
    # difference
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

    y_pred = model.predict([X_test, ds_bops_histogram])

    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ('mean = {}, std = {}'.format(mean, std))

    # print ('Test on large datasets - different distributions')
    # datasets_features_path = 'data/spatial_descriptors/spatial_descriptors_large_datasets_different_distributions.csv'
    # datasets_histograms_path = 'data/histograms/large_datasets_different_distributions'
    # join_results_path = 'data/join_results/join_results_large_datasets_different_distributions_no_bit.csv'
    # features_df = datasets.load_datasets_feature(datasets_features_path)
    # join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data(
    #     features_df,
    #     join_results_path,
    #     datasets_histograms_path,
    #     num_rows, num_columns)
    #
    # X_test = pd.DataFrame.to_numpy(join_data[[i for i in range(num_features)]])
    # y_test = join_data[target]
    #
    # y_pred = model.predict([X_test, ds_all_histogram])
    #
    # print ('r2 score: {}'.format(r2_score(y_test, y_pred)))
    #
    # diff = y_pred.flatten() - y_test
    # percent_diff = (diff / y_test)
    # abs_percent_diff = np.abs(percent_diff)
    # mean = np.mean(abs_percent_diff)
    # std = np.std(abs_percent_diff)
    #
    # print ('mean = {}, std = {}'.format(mean, std))
    #
    # print ('Test on non-aligned small datasets - different distributions')
    # datasets_features_path = 'data/spatial_descriptors/spatial_descriptors_small_datasets_non_aligned.csv'
    # datasets_histograms_path = 'data/histograms/small_datasets_non_aligned_pairs'
    # join_results_path = 'data/join_results/join_results_small_datasets_non_aligned.csv'
    # features_df = datasets.load_datasets_feature(datasets_features_path)
    # join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data2(
    #     features_df,
    #     join_results_path,
    #     datasets_histograms_path,
    #     num_rows, num_columns)
    #
    # X_test = pd.DataFrame.to_numpy(join_data[[i for i in range(num_features)]])
    # y_test = join_data[target]
    #
    # y_pred = model.predict([X_test, ds_all_histogram])
    #
    # print ('r2 score: {}'.format(r2_score(y_test, y_pred)))
    #
    # diff = y_pred.flatten() - y_test
    # percent_diff = (diff / y_test)
    # abs_percent_diff = np.abs(percent_diff)
    # mean = np.mean(abs_percent_diff)
    # std = np.std(abs_percent_diff)
    #
    # print ('mean = {}, std = {}'.format(mean, std))


if __name__ == '__main__':
    main()
