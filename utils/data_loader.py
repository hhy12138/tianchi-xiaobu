import torch
import logging
from torch.utils.data import Dataset

class SimDataset(Dataset):
    def __init__(self,X,Y):
        self.dataset = [(x,y) for x,y in zip(X,Y)]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, left_x, right_x,y, batch_size):

        token_len = max([len(x) for x in left_x]+[len(x) for x in right_x])
        left_text_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        right_text_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        labels = torch.LongTensor(y)
        left_mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)
        right_mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)

        for i, s in enumerate(zip(left_x,right_x,y)):
            left_text_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            right_text_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            left_mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype=torch.uint8)
            right_mask_tokens[i, :len(s[1])] = torch.tensor([1] * len(s[1]), dtype=torch.uint8)

        return left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens

    def collate_fn(self,batch):
        left_x = [data[0][0] for data in batch]
        left_lens = [len(data[0][0]) for data in batch]
        right_x = [data[0][1] for data in batch]
        right_lens = [len(data[0][1]) for data in batch]
        y = [data[1] for data in batch]
        batch_size = len(batch)

        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens = self.get_long_tensor(left_x, right_x,y, batch_size)
        return [left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens,left_lens,right_lens]



