import math

import pandas as pd


def main():
    print ('Compute join results for datasets with same distribution')

    card_factors = [1, 10, 20, 30, 40, 50]
    datasets_cols = ['dataset', 'card']
    result_cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']

    datasets_df = pd.read_csv('../data/join_results_same_distribution/datasets_same_distribution.csv', delimiter=',', header=None, names=datasets_cols)

    join_selectivities_cols = ['cardinality_x_{}'.format(factor) for factor in card_factors]
    join_selectivities_df = pd.DataFrame(columns=join_selectivities_cols)

    dataset1 = None
    dataset2 = None
    for factor in card_factors:
        result_df = pd.read_csv('../data/join_results_same_distribution/join_results_factor_{}.csv'.format(factor), delimiter=',', header=None, names=result_cols)
        result_df = pd.merge(result_df, datasets_df, left_on='dataset1', right_on='dataset')
        result_df = pd.merge(result_df, datasets_df, left_on='dataset2', right_on='dataset')
        cardinality_x = result_df['card_x']
        cardinality_y = result_df['card_y']
        result_size = result_df['result_size']
        join_selectivity = result_size / (cardinality_x * cardinality_y)
        join_selectivity = join_selectivity * math.pow(10, 9)
        result_df = result_df.drop(columns=['mbr_tests', 'duration', 'dataset_x', 'dataset_y'])
        result_df['join_selectivity'] = join_selectivity

        result_df.to_csv('../data/join_results_same_distribution/join_selectivities/join_selectivity_factor_{}.csv'.format(factor))
        dataset1 = result_df['dataset1']
        dataset2 = result_df['dataset2']

        join_selectivities_df['cardinality_x_{}'.format(factor)] = join_selectivity

    dataset1 = [d.replace('_50', '') for d in dataset1]
    dataset2 = [d.replace('_50', '') for d in dataset2]

    join_selectivities_df.insert(0, 'dataset1', dataset1, True)
    join_selectivities_df.insert(1, 'dataset2', dataset2, True)
    join_selectivities_df.to_csv('../data/join_results_same_distribution/join_selectivities/join_selectivities.csv', index=None)


if __name__ == '__main__':
    main()
