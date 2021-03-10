import torch.nn as nn
import torch

class BiLSTM_Cosine(nn.Module):

    def __init__(self,embedding_size,hidden_size,vocab_size,drop_out,linearLayerSize,biGramwd2id,wordvec=None,biGramEmbedding_size=0,bigramLen=0,bigramvec=None):
        super(BiLSTM_Cosine,self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bigramEmbedding = nn.Embedding(bigramLen, biGramEmbedding_size)
        if wordvec is not None:
            init_wordvec = torch.zeros(vocab_size,embedding_size)
            for wd in wordvec.wv.vocab:
                init_wordvec[int(wd),:] = torch.tensor(wordvec.wv[wd])
            self.embedding.weight.data.copy_(torch.Tensor(init_wordvec))
            for para in self.embedding.parameters():
                para.requires_grad = False
        if bigramvec is not None:
            init_biWordvec = torch.zeros(bigramLen, biGramEmbedding_size)
            for wd in bigramvec.wv.vocab:
                key = biGramwd2id[wd]
                init_biWordvec[key,:] = torch.tensor(bigramvec.wv[wd])
            self.bigramEmbedding.weight.data.copy_(torch.Tensor(init_biWordvec))
            for para in self.bigramEmbedding.parameters():
                para.requires_grad = False
        self.bilstm = nn.LSTM(
            input_size=embedding_size+biGramEmbedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True
        )
        self.linearLayer = nn.Linear(hidden_size * 2, linearLayerSize)
        self.leakRelu = nn.LeakyReLU()
        self.criteria = nn.BCELoss()
    def forward(self, left_x,right_x,left_text_bitokens,right_text_bitokens):
        left_oneEmbeddings = self.embedding(left_x)
        right_oneEmbeddings = self.embedding(right_x)
        left_biEmbeddings = self.bigramEmbedding(left_text_bitokens)
        right_biEmbeddings = self.bigramEmbedding(right_text_bitokens)
        left_embeddings = torch.cat((left_oneEmbeddings,left_biEmbeddings),2)
        right_embeddings = torch.cat((right_oneEmbeddings, right_biEmbeddings), 2)
        left_output, (left_h,_) = self.bilstm(left_embeddings)
        right_output, (right_h,_) = self.bilstm(right_embeddings)
        left_h = torch.cat((left_h[-1],left_h[-2]), 1)
        right_h = torch.cat((right_h[-1], right_h[-2]), 1)
        left_linearVec = self.linearLayer(self.leakRelu(left_h))
        right_linearVec = self.linearLayer(self.leakRelu(right_h))
        score = ((torch.cosine_similarity(left_linearVec,right_linearVec)+1)/2)**2
        return score
    def loss(self,left_x,right_x,y,left_text_bitokens,right_text_bitokens):
        score = self.forward(left_x,right_x,left_text_bitokens,right_text_bitokens)
        self.criteria.weight = y*2+1
        loss = self.criteria(score, y.to(torch.float32))
        return loss