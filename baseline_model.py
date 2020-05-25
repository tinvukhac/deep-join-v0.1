import datasets

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def main():
    print ('Train and test join data using baseline model')
    features_df = datasets.load_datasets_feature('data/uniform_datasets_features.csv')
    join_data, ds1_histograms, ds2_histograms = datasets.load_join_data(features_df, 'data/uniform_result_size.csv')
    train_attributes, test_attributes = train_test_split(join_data, test_size=0.25, random_state=42)

    X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(1, 13)]])
    X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(1, 13)]])
    y_train = train_attributes[0]
    y_test = test_attributes[0]

    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))


if __name__ == '__main__':
    main()
