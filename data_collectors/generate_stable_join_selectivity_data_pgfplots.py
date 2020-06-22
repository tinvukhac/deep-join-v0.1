def main():
    print ('Generate data to copy to pgfplots')
    count = 5
    query_ids = ['group {}'.format(i) for i in range(1, count + 1)]

    print (','.join(query_ids))

    f = open('../data/bak/join_results_same_distribution_diff_width_height/join_selectivities/join_selectivities.csv')

    i = 0
    line = f.readline()
    line = f.readline()
    while line and i < count:
        i += 1
        data = line.strip().split(',')
        mean = data[8]
        std = data[9]
        print ('(group {},{}) +- ({},{})'.format(i, mean, std, std))

        line = f.readline()
    f.close()


if __name__ == '__main__':
    main()
