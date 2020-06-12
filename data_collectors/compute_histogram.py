import numpy as np


def extract_histogram(input_filename, output_filename, num_rows, num_columns):
    hist = np.zeros((num_rows, num_columns))
    input_f = open(input_filename)

    line = input_f.readline()
    line = input_f.readline()
    while line:
        data = line.strip().split('\t')
        column = int(data[0])
        row = int(data[1])
        freq = int(data[3])
        hist[row][column] = freq

        line = input_f.readline()

    np.savetxt(output_filename, hist.astype(int), fmt='%i', delimiter=',')

    input_f.close()


def extract_histograms():
    histogram_dirs = ['32x32']
    # f = open('../data/large_datasets.csv')
    f = open('../data/all_datasets.txt')
    lines = f.readlines()
    filenames = [line.strip() for line in lines]

    for histogram_dir in histogram_dirs:
        data = histogram_dir.split('x')
        num_rows = int(data[0])
        num_columns = int(data[1])
        input_dir = '../data/data_nonaligned/histograms/histograms_union_mbr/small_datasets/{}'.format(histogram_dir)
        output_dir = '../data/data_nonaligned/histograms/small_datasets/{}'.format(histogram_dir)
        for filename in filenames:
            extract_histogram('{}/{}'.format(input_dir, filename), '{}/{}'.format(output_dir, filename), num_rows, num_columns)


def shrink_histograms(num_rows, num_columns):
    input_dir = '../data/data_aligned/histograms/small_datasets/{}x{}'.format(num_rows, num_columns)
    output_dir = '../data/data_aligned/histograms/small_datasets/{}x{}'.format(num_rows / 2, num_columns / 2)

    f = open('../data/all_datasets.txt')
    lines = f.readlines()
    filenames = [line.strip() for line in lines]

    for dataset in filenames:
        hist_input_filename = '{}/{}'.format(input_dir, dataset)
        hist_output_filename = '{}/{}'.format(output_dir, dataset)

        hist_input = np.genfromtxt(hist_input_filename, delimiter=',')
        hist_output = np.zeros((num_rows / 2, num_columns / 2))

        for i in range(hist_output.shape[0]):
            for j in range(hist_output.shape[1]):
                hist_output[i, j] = hist_input[2*i, 2*j] + hist_input[2*i + 1, 2*j] + hist_input[2*i, 2*j + 1] + hist_input[2*i + 1, 2*j + 1]

        np.savetxt(hist_output_filename, hist_output.astype(int), fmt='%i', delimiter=',')


def main():
    print('Compute histogram')
    # extract_histograms()
    shrink_histograms(32, 32)

if __name__ == '__main__':
    main()
