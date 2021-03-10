import torch
from torch.utils.data import DataLoader
from utils.data_loader import SimDataset
from utils.read import read
from utils.dictionary import buildBiGramDictionary

from config import config
import logging
from tqdm import tqdm

from sklearn.metrics import accuracy_score
import numpy as np

def epoch_train(train_loader, model, optimizer, scheduler, epoch,device):
    model.train()
    train_loss = 0.0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens, left_lens, right_lens,left_text_bitokens,right_text_bitokens = batch_samples
        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens, left_lens, right_lens,left_text_bitokens,right_text_bitokens = left_text_tokens.to(device), right_text_tokens.to(device), labels.to(device), left_mask_tokens.to(device), right_mask_tokens.to(device), left_lens, right_lens,left_text_bitokens.to(device),right_text_bitokens.to(device)
        model.zero_grad()
        loss = model.loss(left_text_tokens, right_text_tokens,labels,left_text_bitokens,right_text_bitokens)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    print("epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler,device):
    best_accuracy = 0.0
    for epoch in range(1, config.epoch_num + 1):
        epoch_train(train_loader, model, optimizer, scheduler,epoch,device)
        with torch.no_grad():
            metric = dev(dev_loader, model,device)
            print("epoch: {}, dev loss: {}, accuracy: {}".format(epoch, metric['loss'],metric['accuracy']))
            if metric['accuracy'] > best_accuracy:
                torch.save(model, config.model_dir)
    logging.info("Training Finished!")

def dev(data_loader, model, device,mode='dev'):
    model.eval()
    true_y = []
    pred_y = []
    dev_losses = 0
    for idx, batch_samples in enumerate(data_loader):
        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens, left_lens, right_lens,left_text_bitokens,right_text_bitokens = batch_samples
        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens, left_lens, right_lens,left_text_bitokens,right_text_bitokens = left_text_tokens.to(device), right_text_tokens.to(device), labels.to(device), left_mask_tokens.to(device), right_mask_tokens.to(device), left_lens, right_lens,left_text_bitokens.to(device),right_text_bitokens.to(device)
        y_pred= model.forward(left_text_tokens, right_text_tokens,left_text_bitokens,right_text_bitokens).cpu()
        loss = model.loss(left_text_tokens, right_text_tokens,labels,left_text_bitokens,right_text_bitokens)
        pred_y.extend(y_pred.round())
        true_y.extend(labels.cpu())
        dev_losses += loss.item()

    metrix = {}
    accuracy = accuracy_score(true_y,pred_y)
    metrix['accuracy'] = accuracy
    dev_loss = float(dev_losses) / len(data_loader)
    metrix['loss'] = dev_loss
    return metrix

def test():
    X_train, _ = read(config.train_dir, mode='test')
    X_test, _ = read(config.test_dir, mode='test')
    X = X_train + X_test
    bigramWd2id, bigramId2wd, bigramLen = buildBiGramDictionary(X)
    y_preds = []
    X,Y = read(config.test_dir,mode='test')
    Y = [0]*len(X)
    test_dataset = SimDataset(X,Y,bigramWd2id)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=test_dataset.collate_fn)
    model = torch.load(config.model_dir,map_location=torch.device('cpu'))
    for idx, batch_samples in enumerate(tqdm(test_loader)):
        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens, left_lens, right_lens,left_text_bitokens,right_text_bitokens = batch_samples
        y_pred= model.forward(left_text_tokens, right_text_tokens,left_text_bitokens,right_text_bitokens)
        y_preds.extend(y_pred)
    with open(config.output_dir,'w') as f:
        for i in y_preds:
            f.write(str(i.item())+'\n')
test()
###################
def test1(model_dir,output_dir):
    y_preds = []
    X,Y = read(config.test_dir,mode='test')
    Y = [0]*len(X)
    test_dataset = SimDataset(X,Y)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=test_dataset.collate_fn)
    print(model_dir)
    model = torch.load(model_dir,map_location=torch.device('cpu'))
    for idx, batch_samples in enumerate(tqdm(test_loader)):
        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens, left_lens, right_lens = batch_samples
        y_pred= model.forward(left_text_tokens, right_text_tokens)
        y_preds.extend(y_pred)
    with open(output_dir,'w') as f:
        for i in y_preds:
            f.write(str(i.item())+'\n')

# import os
# for root,dir,files in os.walk('history_models'):
#     for file in files:
#         if file.endswith('.pth'):
#             output_dir = "output/{}.txt".format(file)
#             test1(root+'/'+file,output_dir)
# # scores = [0]*25000
# # for root,dir,files in os.walk('output'):
# #     for file in files:
# #         if file.endswith('txt'):
# #             with open(root+'/'+file,'r') as f:
# #                 for id,line in enumerate(f.readlines()):
# #                     line = float(line.strip())
# #                     scores[id]+=(0.05*line)
# #
# # with open('output/result.txt','w') as f:
# #     for score in scores:
# #         f.write(str(score)+'\n')