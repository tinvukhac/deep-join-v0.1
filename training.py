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


def main():
    print ('Training the join cardinality estimator')
    features_df = datasets.load_datasets_feature('data/datasets_features.csv')
    join_data, ds1_histograms, ds2_histograms = datasets.load_join_data(features_df, 'data/result_size.csv')
    train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test = train_test_split(
        join_data, ds1_histograms, ds2_histograms, test_size=0.25, random_state=42)

    X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(1, 13)]])
    X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(1, 13)]])
    y_train = train_attributes[0]
    y_test = test_attributes[0]

    mlp = models.create_mlp(X_train.shape[1], regress=False)
    num_rows, num_columns = 16, 16
    cnn1 = models.create_cnn(num_rows, num_columns, 1, regress=False)
    cnn2 = models.create_cnn(num_rows, num_columns, 1, regress=False)

    combined_input = concatenate([mlp.output, cnn1.output, cnn2.output])

    x = Dense(4, activation="relu")(combined_input)
    x = Dense(1, activation="linear")(x)

    model = Model(inputs=[mlp.input, cnn1.input, cnn2.input], outputs=x)

    EPOCHS = 200
    LR = 1e-3
    # opt = Adam(lr=1e-4, decay=1e-4 / 200)
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # train the model
    print("[INFO] training model...")
    model.fit(
        [X_train, ds1_histograms_train, ds2_histograms_train], y_train,
        validation_data=([X_test, ds1_histograms_test, ds2_histograms_test], y_test),
        epochs=EPOCHS, batch_size=32)

    y_pred = model.predict([X_test, ds1_histograms_test, ds2_histograms_test])
    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test) * 100
    abs_percent_diff = np.abs(percent_diff)

    # compute the mean and standard deviation of the absolute percentage
    # difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    print ("Synthetic dataset:")
    print ('mean = {}, std = {}'.format(mean, std))


if __name__ == '__main__':
    main()
