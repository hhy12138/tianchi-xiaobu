def read(path,mode='train'):
    X = []
    Y = None
    if mode == 'train':
        Y = []
    with open(path,'r') as f:
        for line in f.readlines():
            data = line.split('\t')
            left,right = data[0].strip(),data[1].strip()
            left = [int(i) for i in left.split(' ')]
            right = [int(i) for i in right.split(' ')]
            X.append([left,right])
            if mode == 'train':
                label = int(data[2].strip())
                Y.append(label)
    return X,Y



if __name__ == "__main__":
    X,Y = read("../data/gaiic_track3_round1_testA_20210228.tsv",mode='test')