from sklearn.model_selection import train_test_split
from config import config
from utils.read import read
from utils.data_loader import SimDataset
from torch.utils.data import DataLoader
from utils.data_loader import SimDataset
from models.BiLSTM_Cosine import BiLSTM_Cosine
from torch.optim.lr_scheduler import StepLR
from torch import optim
from train import train
from gensim.models import Word2Vec


def dev_split(dataset_dir):
    X,Y = read(dataset_dir,mode='train')
    x_train, x_dev, y_train, y_dev = train_test_split(X, Y, test_size=config.dev_split_size, random_state=0)
    return x_train, x_dev, y_train, y_dev

def simple_run():
    x_train, x_dev, y_train, y_dev = dev_split(config.train_dir)
    run(x_train, x_dev, y_train, y_dev)
    #run(x_train, x_train, y_train, y_train)

def run(x_train, x_dev, y_train, y_dev):
    train_dataset = SimDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)

    dev_dataset = SimDataset(x_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=dev_dataset.collate_fn)
    if config.onegramvec != '':
        vec = Word2Vec.load(config.onegramvec)
    model = BiLSTM_Cosine(embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       drop_out=config.drop_out,
                       vocab_size=config.vocab_size,
                          linearLayerSize=config.linearLayerSize,
                          wordvec=vec)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
    train(train_loader, dev_loader, model, optimizer, scheduler)


if __name__ == "__main__":
    simple_run()
