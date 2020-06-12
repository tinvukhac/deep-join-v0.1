from keras import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import concatenate

import datasets
import models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import math


def main():
    print ('Training the join cardinality estimator')
    num_rows, num_columns = 16, 16
    # features_df = datasets.load_datasets_feature('data/data_aligned/aligned_small_datasets_features.csv')
    # join_data, ds1_histograms, ds2_histograms, ds_all_histogram = datasets.load_join_data(features_df,
    #                                                                                       'data/data_aligned/join_results_small_datasets.csv',
    #                                                                                       'data/data_aligned/histograms/small_datasets',
    #                                                                                       num_rows, num_columns)

    features_df_small = datasets.load_datasets_feature('data/data_nonaligned/nonaligned_small_datasets_features.csv')
    join_data_small, ds1_histograms_small, ds2_histograms_small, ds_all_histogram_small = datasets.load_join_data(features_df_small,
                                                                                          'data/data_nonaligned/join_results_small_datasets.csv',
                                                                                          'data/data_nonaligned/histograms/small_datasets',
                                                                                          num_rows, num_columns)

    # features_df_small_vary = datasets.load_datasets_feature('data/data_nonaligned/nonaligned_small_datasets_vary_features.csv')
    # join_data_small_vary, ds1_histograms_small_vary, ds2_histograms_small_vary, ds_all_histogram_small_vary = datasets.load_join_data(features_df_small_vary,
    #                                                                                       'data/data_nonaligned/join_results_small_datasets_vary_non_zero.csv',
    #                                                                                       'data/data_nonaligned/histograms/small_datasets_vary',
    #                                                                                       num_rows, num_columns)

    train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_all_histogram_train, ds_all_histogram_test = train_test_split(
        join_data_small, ds1_histograms_small, ds2_histograms_small, ds_all_histogram_small, test_size=0.25,
        random_state=42)

    # join_data = pd.concat([join_data_small, join_data_small_vary])
    # ds1_histograms = np.concatenate((ds1_histograms_small, ds1_histograms_small_vary), axis=0)
    # ds2_histograms = np.concatenate((ds2_histograms_small, ds2_histograms_small_vary), axis=0)
    # ds_all_histogram = np.concatenate((ds_all_histogram_small, ds_all_histogram_small_vary), axis=0)
    #
    # print (join_data.shape)
    #
    # train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_all_histogram_train, ds_all_histogram_test = train_test_split(
    #     join_data, ds1_histograms, ds2_histograms, ds_all_histogram, test_size=0.25, random_state=42)

    num_features = len(train_attributes.columns) - 7
    # print (join_data)
    X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(num_features)]])
    X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(num_features)]])
    y_train = train_attributes['join_selectivity']
    y_test = test_attributes['join_selectivity']
    # y_train = train_attributes['result_size']
    # y_test = test_attributes['result_size']

    mlp = models.create_mlp(X_train.shape[1], regress=False)
    cnn1 = models.create_cnn(num_rows, num_columns, 2, regress=False)
    # cnn2 = models.create_cnn(num_rows, num_columns, 1, regress=False)

    # combined_input = concatenate([mlp.output, cnn1.output, cnn2.output])
    combined_input = concatenate([mlp.output, cnn1.output])

    x = Dense(4, activation="relu")(combined_input)
    x = Dense(1, activation="linear")(x)

    # model = Model(inputs=[mlp.input, cnn1.input, cnn2.input], outputs=x)
    model = Model(inputs=[mlp.input, cnn1.input], outputs=x)

    EPOCHS = 50
    LR = 1e-2
    # opt = Adam(lr=1e-4, decay=1e-4 / 200)
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    print (model.summary())

    # train the model
    print("[INFO] training model...")
    # model.fit(
    #     [X_train, ds1_histograms_train, ds2_histograms_train], y_train,
    #     validation_data=([X_test, ds1_histograms_test, ds2_histograms_test], y_test),
    #     epochs=EPOCHS, batch_size=128)
    model.fit(
        [X_train, ds_all_histogram_train], y_train,
        validation_data=([X_test, ds_all_histogram_test], y_test),
        epochs=EPOCHS, batch_size=32)

    y_pred = model.predict([X_test, ds_all_histogram_test])
    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ("Small dataset:")
    print ('mean = {}, std = {}'.format(mean, std))


if __name__ == '__main__':
    main()
