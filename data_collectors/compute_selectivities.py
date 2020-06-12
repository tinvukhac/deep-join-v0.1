import datasets

import numpy as np
import pandas as pd


def main():
    print ('Compute selectivities in each grid cell')

    num_rows = 16
    num_columns = 16

    dataset1 = 'diagonal_001.csv'
    dataset2 = 'gaussian_001.csv'

    # input_f = open('../data/histograms/{}x{}/diagonal_001.csv'.format(num_rows, num_columns))
    normalized_hist1, hist1 = datasets.load_histogram('../data/histogram_large_values', num_rows, num_columns, dataset1)
    normalized_hist2, hist2 = datasets.load_histogram('../data/histogram_large_values', num_rows, num_columns, dataset2)

    # print (hist1)
    # print (hist2)

    cols = ['count']
    result_counts_df = pd.read_csv('../data/result_count.csv', delimiter=',', header=None, names=cols)
    # print (result_counts_df)
    result_counts = result_counts_df['count'].to_numpy().reshape((num_rows, num_columns))
    result_counts = np.array(result_counts, dtype=float)

    # print (result_counts)

    bops = hist1 * hist2
    bops = bops.reshape((16, 16))
    bops = np.array(bops, dtype=float)
    # print (bops)
    selectivities = np.divide(result_counts, bops, out=np.zeros_like(result_counts), where=bops != 0)
    # print (selectivities)
    np.savetxt('../data/selectivities_large.csv', selectivities, delimiter=',')

    print (np.mean(selectivities))
    print (np.std(selectivities))

    selectivities_1dim = selectivities.reshape((256, 1))
    selectivities_1dim = selectivities_1dim[selectivities_1dim != 0]
    print (selectivities_1dim)

    print (np.mean(selectivities_1dim))
    print (np.std(selectivities_1dim))



if __name__ == '__main__':
    main()
