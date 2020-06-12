def main():
    print ('Filter data')
    input_f = open('../data/data_nonaligned/join_results_small_datasets_vary.csv')
    output_f = open('../data/data_nonaligned/join_results_small_datasets_vary_non_zero.csv', 'w')

    lines = input_f.readlines()

    for line in lines:
        data = line.strip().split(',')
        result_size = int(data[2])
        if result_size != 0:
            output_f.writelines(line)

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
