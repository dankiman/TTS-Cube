import sys

if len(sys.argv) < 4:
    print("Usage split_g2p_corpus.py <input file> <train file> <dev file>")
else:
    input_file = open(sys.argv[1])
    train_file = open(sys.argv[2], 'w')
    dev_file = open(sys.argv[3], 'w')
    cnt = 0
    for line in input_file.readlines():
        cnt += 1
        if cnt % 10 == 0:
            t_file = dev_file
        else:
            t_file = train_file

        t_file.write(line)
    train_file.close()
    dev_file.close()
