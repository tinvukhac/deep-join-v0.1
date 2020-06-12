def main():
    print ('Filter data')
    input_f = open('../data/data_aligned/join_results_large_datasets.csv')
    output_f = open('../data/data_aligned/join_results_large_datasets_noparcel.csv', 'w')

    lines = input_f.readlines()

    for line in lines:
        # data = line.strip().split(',')
        # result_size = int(data[2])
        if 'parcel' not in line:
            output_f.writelines(line)

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
