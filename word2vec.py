from utils.read import read
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
def generateOneGram():
    trainData,_ = read('data/gaiic_track3_round1_train_20210228.tsv',mode='test')
    testData,_ = read('data/gaiic_track3_round1_testA_20210228.tsv',mode='test')
    with open("tmpData/one-gramTmp.txt",'w') as f:
        for x in trainData+testData:
            f.write(' '.join([str(i) for i in x[0]])+'\n')
            f.write(' '.join([str(i) for i in x[1]]) + '\n')

def trainOneGram():
    model = Word2Vec(LineSentence("tmpData/one-gramTmp.txt"), sg=1,size=100, window=3, min_count=1, workers=multiprocessing.cpu_count())
    model.save("vec/one-gram.vec")
if __name__=='__main__':

    model = Word2Vec.load('vec/one-gram.vec')
    cnt = 0
    max_n = 0
    for i in model.wv.vocab:
        print(i)
        cnt+=1
        if int(i) > max_n:
            max_n = int(i)
    print(cnt,max_n)
