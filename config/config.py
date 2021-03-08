import os
data_dir = os.getcwd()+'/data/'
train_dir = data_dir+'gaiic_track3_round1_train_20210228.tsv'
test_dir = data_dir+'gaiic_track3_round1_testA_20210228.tsv'

gpu = ''

dev_split_size = 0.1
batch_size = 32

linearLayerSize=16

hidden_size = 20

drop_out = 0.5

vocab_size = 21963

lr = 1e-3
lr_step = 5
lr_gamma = 0.8
epoch_num = 100
betas = (0.9, 0.999)
embedding_size = 100

onegramvec = "vec/one-gram.vec"