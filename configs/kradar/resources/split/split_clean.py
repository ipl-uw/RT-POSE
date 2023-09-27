LIST_BAD = [27,28,29,35, 36,37, 47, 53,57,58]
LIST_OK = [1,2,3, 4, 6, 26, 30,31,32,33,34,38,39,40,41,42, 43,44,45, 46, 48,49,50, 51, 52, 54,55,56]
LIST_GOOD = [5,7,8,9,10,11,12,13, 14, 15,16,17,18,19,20,21,22, 23, 24, 25]

if __name__ == '__main__':
    with open('train.txt', 'r') as in_file:
        lines = in_file.readlines()
    new_lines = []
    for line in lines:
        seq = line.split(',')[0]
        if int(seq) not in LIST_BAD:
            new_lines.append(line)
    with open('train_clean.txt', 'w') as out_file:
        out_file.writelines(new_lines)