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
def generateBiGram():
    trainData, _ = read('data/gaiic_track3_round1_train_20210228.tsv', mode='test')
    testData, _ = read('data/gaiic_track3_round1_testA_20210228.tsv', mode='test')
    with open("tmpData/bi-gramTmp.txt", 'w') as f:
        for x in trainData + testData:
            line = ""
            for id,_ in enumerate(x[0]):
                if id < len(x[0])-1:
                    line+='{}|{} '.format(str(x[0][id]),str(x[0][id+1]))
                else:
                    line += '{}|end'.format(str(x[0][id]))
            f.write(line+'\n')
            line = ""
            for id, _ in enumerate(x[1]):
                if id < len(x[1]) - 1:
                    line += '{}|{} '.format(str(x[1][id]), str(x[1][id + 1]))
                else:
                    line += '{}|end'.format(str(x[1][id]))
            f.write(line + '\n')

def trainOneGram():
    model = Word2Vec(LineSentence("tmpData/one-gramTmp.txt"), sg=1,size=100, window=5, min_count=1, iter=10,workers=multiprocessing.cpu_count())
    model.save("vec/one-gram.vec")
def trainBiGram():
    model = Word2Vec(LineSentence("tmpData/bi-gramTmp.txt"), sg=1,size=100, window=5, min_count=1, iter=10,workers=multiprocessing.cpu_count())
    model.save("vec/bi-gram.vec")
if __name__=='__main__':
    # generateOneGram()
    # trainOneGram()
    model = Word2Vec.load('vec/bi-gram.vec')
    print(model.wv['1|2'])
