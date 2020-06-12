def main():
    print ('Filter data')
    input_f = open('../data/result_mbr_all.csv')
    output_f = open('../data/result_mbr.csv', 'w')

    lines = input_f.readlines()

    for line in lines:
        if 'error' not in line:
            output_f.writelines(line)

    output_f.close()
    input_f.close()


if __name__ == '__main__':
    main()
