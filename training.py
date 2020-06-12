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
    features_df = datasets.load_datasets_feature('data/datasets_features.csv')
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram = datasets.load_join_data(features_df,
                                                                                              'data/result_size.csv',
                                                                                            'data/histogram_values',
                                                                                              num_rows, num_columns)
    train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_all_histogram_train, ds_all_histogram_test = train_test_split(
        join_data, ds1_histograms, ds2_histograms, ds_all_histogram, test_size=0.25, random_state=42)

    print (ds_all_histogram.shape)

    num_features = len(join_data.columns) - 7
    print (join_data)
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

    np.savetxt('prediction.csv', [y_test, y_pred.flatten()], delimiter=',')

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ("Small dataset:")
    print ('mean = {}, std = {}'.format(mean, std))

    y_test_array = np.asarray(y_test)
    y_pred_array = np.asarray(y_pred)

    y_test_array = y_test_array.reshape((len(y_test_array), 1))
    y_pred_array = y_pred_array.reshape((len(y_pred_array), 1))

    np.savetxt('proposed_model_prediction.csv', np.concatenate([y_test_array, y_pred_array], axis=1), fmt='%.3f',
               delimiter=',')

    # Test on data points for large file
    features_df_large = datasets.load_datasets_feature('data/datasets_features.bak3.csv')
    join_data_large, ds1_histograms_large, ds2_histograms_large, ds_all_histogram_large = datasets.load_join_data(features_df_large,
                                                                                          'data/result_size_large.csv',
                                                                                          'data/histogram_values.bak2',
                                                                                          num_rows, num_columns)
    X_test = pd.DataFrame.to_numpy(join_data_large[[i for i in range(num_features)]])
    y_test = join_data_large['result_size']

    cardinality_x = join_data_large['cardinality_x']
    cardinality_y = join_data_large['cardinality_y']
    y_pred_selectivity = model.predict([X_test, ds_all_histogram_large])
    y_pred = y_pred_selectivity * cardinality_x * cardinality_y / math.pow(10, 9)
    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    # np.savetxt('prediction.csv', [y_test, y_pred.flatten()], delimiter=',')

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ("Large dataset:")
    print ('mean = {}, std = {}'.format(mean, std))


if __name__ == '__main__':
    main()
