import torch
import logging
from torch.utils.data import Dataset
from utils.dictionary import getBiWds
import numpy as np
import math

class SimDataset(Dataset):
    def __init__(self,X,Y,bigramWd2id):
        self.dataset = [(x,y) for x,y in zip(X,Y)]
        self.bigramWd2id = bigramWd2id

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def editDistance(self,left_x,right_x):
        len1 = len(left_x)
        len2 = len(right_x)
        dp = np.zeros((len1 + 1, len2 + 1))
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                delta = 0 if left_x[i - 1] == right_x[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
        return dp[len1][len2]

    def sameWds(self,left_x,right_x):
        return len(set(left_x).intersection(set(right_x)))


    def get_long_tensor(self, left_x, right_x,y, batch_size,leftBiIds,rightBiIds,lenDiffRate,editDistanceRate,sameWdsRate):

        token_len = max([len(x) for x in left_x]+[len(x) for x in right_x])
        left_text_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        right_text_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        left_text_bitokens = torch.LongTensor(batch_size, token_len).fill_(0)
        right_text_bitokens = torch.LongTensor(batch_size, token_len).fill_(0)
        labels = torch.LongTensor(y)
        left_mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)
        right_mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)
        lenDiffRate, editDistanceRate, sameWdsRate = torch.tensor(lenDiffRate),torch.tensor(editDistanceRate),torch.tensor(sameWdsRate)

        for i, s in enumerate(zip(left_x,right_x,leftBiIds,rightBiIds,y)):
            left_text_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            right_text_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            left_text_bitokens[i, :len(s[2])] = torch.LongTensor(s[2])
            right_text_bitokens[i, :len(s[3])] = torch.LongTensor(s[3])
            left_mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype=torch.uint8)
            right_mask_tokens[i, :len(s[1])] = torch.tensor([1] * len(s[1]), dtype=torch.uint8)

        return left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens,left_text_bitokens,right_text_bitokens,lenDiffRate,editDistanceRate,sameWdsRate

    def collate_fn(self,batch):
        left_x = [data[0][0] for data in batch]
        left_lens = [len(data[0][0]) for data in batch]
        right_x = [data[0][1] for data in batch]
        leftBiWds = [getBiWds(x) for x in left_x]
        rightBiWds = [getBiWds(x) for x in right_x]
        leftBiIds = [[self.bigramWd2id[wd] for wd in sentence] for sentence in leftBiWds]
        rightBiIds = [[self.bigramWd2id[wd] for wd in sentence] for sentence in rightBiWds]
        right_lens = [len(data[0][1]) for data in batch]
        y = [data[1] for data in batch]
        batch_size = len(batch)

        lenDiffRate = [abs(left_len-right_len)/(left_len+right_len) for left_len,right_len in zip(left_lens,right_lens)]
        editDistanceRate = [self.editDistance(left,right)/(len(left)+len(right)) for left,right in zip(left_x,right_x)]
        sameWdsRate = [self.sameWds(left,right)/(len(left)+len(right)) for left,right in zip(left_x,right_x)]

        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens,left_text_bitokens,right_text_bitokens,lenDiffRate,editDistanceRate,sameWdsRate = self.get_long_tensor(left_x, right_x,y, batch_size,leftBiIds,rightBiIds,lenDiffRate,editDistanceRate,sameWdsRate)
        return [left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens,left_lens,right_lens,left_text_bitokens,right_text_bitokens,lenDiffRate,editDistanceRate,sameWdsRate]



