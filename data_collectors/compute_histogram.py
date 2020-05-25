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
    histogram_dirs = ['16x16', '32x32']
    f = open('../data/uniform_datasets.txt')
    lines = f.readlines()
    filenames = [line.strip() for line in lines]

    for histogram_dir in histogram_dirs:
        data = histogram_dir.split('x')
        num_rows = int(data[0])
        num_columns = int(data[1])
        input_dir = '../data/histograms_uniform/{}'.format(histogram_dir)
        output_dir = '../data/histogram_uniform_values/{}'.format(histogram_dir)
        for filename in filenames:
            extract_histogram('{}/{}.csv'.format(input_dir, filename), '{}/{}.csv'.format(output_dir, filename), num_rows, num_columns)


def main():
    print('Compute histogram')
    extract_histograms()


if __name__ == '__main__':
    main()
