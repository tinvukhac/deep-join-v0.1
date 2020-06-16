def main():
    print ('Filter data')
    input_f = open('../data/join_results/join_results_large_datasets_different_distributions.csv')
    output_f = open('../data/join_results/join_results_large_datasets_different_distributions_no_bit.csv', 'w')

    lines = input_f.readlines()

    for line in lines:
        # data = line.strip().split(',')
        # result_size = int(data[2])
        if 'bit' not in line:
            output_f.writelines(line)

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
