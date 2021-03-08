import torch.nn as nn
import torch

class BiLSTM_Cosine(nn.Module):

    def __init__(self,embedding_size,hidden_size,vocab_size,drop_out,linearLayerSize,wordvec=None):
        super(BiLSTM_Cosine,self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if wordvec is not None:
            init_wordvec = torch.zeros(vocab_size,embedding_size)
            for wd in wordvec.wv.vocab:
                init_wordvec[int(wd),:] = torch.tensor(wordvec.wv[wd])
            self.embedding.weight.data.copy_(torch.Tensor(init_wordvec))
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True
        )
        self.linearLayer = nn.Linear(hidden_size * 2, linearLayerSize)
        self.criteria = nn.BCELoss()
    def forward(self, left_x,right_x,y):
        left_embeddings = self.embedding(left_x)
        right_embeddings = self.embedding(right_x)
        left_output, (left_h,_) = self.bilstm(left_embeddings)
        right_output, (right_h,_) = self.bilstm(right_embeddings)
        left_h = torch.cat((left_h[-1],left_h[-2]), 1)
        right_h = torch.cat((right_h[-1], right_h[-2]), 1)
        left_linearVec = self.linearLayer(left_h)
        right_linearVec = self.linearLayer(right_h)
        score = (torch.cosine_similarity(left_linearVec,right_linearVec)+1)/2
        loss = self.criteria(score,y.to(torch.float32))
        return score,loss