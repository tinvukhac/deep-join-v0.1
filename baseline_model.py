import datasets

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def main():
    print ('Train and test join data using baseline model')
    features_df = datasets.load_datasets_feature('data/datasets_features.csv')
    join_data, ds1_histograms, ds2_histograms = datasets.load_join_data(features_df, 'data/result_size.csv')
    train_attributes, test_attributes = train_test_split(join_data, test_size=0.25, random_state=42)

    X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(18)]])
    X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(18)]])
    y_train = train_attributes['result_size']
    y_test = test_attributes['result_size']

    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
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

    y_test_array = np.asarray(y_test)
    y_pred_array = np.asarray(y_pred)

    y_test_array = y_test_array.reshape((len(y_test_array), 1))
    y_pred_array = y_test_array.reshape((len(y_pred_array), 1))

    print (np.sum(y_test_array - y_pred_array))

    np.savetxt('baseline_prediction_all.csv', np.concatenate([y_test_array, y_pred_array], axis=1), fmt='%.3f', delimiter=',')


if __name__ == '__main__':
    main()
