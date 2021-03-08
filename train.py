import torch
from torch.utils.data import DataLoader

from config import config
import logging
from tqdm import tqdm

from sklearn.metrics import accuracy_score
import numpy as np

def epoch_train(train_loader, model, optimizer, scheduler, epoch):
    model.train()
    train_loss = 0.0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens, left_lens, right_lens = batch_samples
        model.zero_grad()
        scores,loss = model.forward(left_text_tokens, right_text_tokens,labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    train_loss = float(train_loss) / len(train_loader)
    print("epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler):
    best_accuracy = 0.0
    for epoch in range(1, config.epoch_num + 1):
        epoch_train(train_loader, model, optimizer, scheduler,epoch)
        with torch.no_grad():
            metric = dev(dev_loader, model)
            print("epoch: {}, dev loss: {}, accuracy: {}".format(epoch, metric['loss'],metric['accuracy']))
            if metric['accuracy'] > best_accuracy:
                torch.save(model, config.model_dir)
    logging.info("Training Finished!")

def dev(data_loader, model, mode='dev'):
    model.eval()
    true_y = []
    pred_y = []
    dev_losses = 0
    for idx, batch_samples in enumerate(data_loader):
        left_text_tokens, right_text_tokens, labels, left_mask_tokens, right_mask_tokens, left_lens, right_lens = batch_samples

        y_pred,loss = model.forward(left_text_tokens, right_text_tokens,labels)
        pred_y.extend(y_pred.round())
        true_y.extend(labels)
        dev_losses += loss.item()

    metrix = {}
    accuracy = accuracy_score(true_y,pred_y)
    metrix['accuracy'] = accuracy
    dev_loss = float(dev_losses) / len(data_loader)
    metrix['loss'] = dev_loss
    return metrix