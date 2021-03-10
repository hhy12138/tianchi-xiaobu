import os
data_dir = os.getcwd()+'/data/'
train_dir = data_dir+'gaiic_track3_round1_train_20210228.tsv'
test_dir = data_dir+'gaiic_track3_round1_testA_20210228.tsv'
model_dir = 'history_models/' + 'model.pth'
output_dir = 'output/' + 'result10.txt'
gpu = '0'

dev_split_size = 0.4
batch_size = 128

linearLayerSize=16

hidden_size = 32

drop_out = 0.01

vocab_size = 21963

lr = 3e-3
lr_step = 5
lr_gamma = 0.8
epoch_num = 12
betas = (0.9, 0.999)
embedding_size = 100
biGramEmbedding_size = 100

onegramvec = "vec/one-gram.vec"
bigramvec = "vec/bi-gram.vec"