import sys

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from utility import *

class Tagging_Dataset(Dataset):
    def __init__(self, data):
        self.windows, self.tags = data

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        tag = self.tags[idx]
        return torch.tensor(window), torch.tensor(tag)
    
class tagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_layer, target_size, dropout_p, pre_trained_embeddings=None, WINDOW_SIZE=5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * WINDOW_SIZE, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, target_size)
        self.activation = nn.Tanh()
        self.dropout1 = nn.Dropout(p=dropout_p)
        # part 3 pretrained vocab
        self.WINDOW_SIZE = WINDOW_SIZE

    def forward(self, sentence):
        x = self.word_embeddings(sentence).view(-1, self.embedding_dim * self.WINDOW_SIZE)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x
    
def accuracy(pred, output, mission='NER'):
    corrects = 0
    if mission.upper() == 'NER':
        for p, o in zip(pred, output):
            if (pred==output) & (output==vocab.stoi['O']):
                continue
            else:
                corrects += 1
        return corrects/len(pred)
                



def epoch_train(model, optimizer, criterion, train_loader, val_loader, mission='NER'):
    running_loss = 0
    for windows, tags in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(windows)
        loss = criterion(output, tags)
        running_loss += loss
        loss.backward()
        optimizer.step()

        pred = torch.argmax(output)
        accuracy(mission, pred, output)

    return running_loss/len(train_loader)

def evaluate(model, criterion, val_loader):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for windows, tags in (val_loader):
            output = model(windows)
            loss = criterion(output, tags)
            running_loss += loss
        return running_loss/len(val_loader)


def train(model, optimizer, criterion, nepochs, train_loader, val_loader, mission='NER'):
    model.train()
    train_losses, val_losses = [], []
    for e in range(nepochs):
        loss = epoch_train(model, optimizer, criterion, train_loader, val_loader, mission='NER')
        val_loss = evaluate(model, criterion, val_loader)
        print(f'epoch {e+1}: train loss: {loss}. val loss: {val_loss}')
        train_losses += [loss]
        val_losses += [val_loss]

    return train_losses, val_losses






train_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, train_data))
train_data_loader = DataLoader(train_dataset,
                                batch_size=128, shuffle=True)



dev_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, dev_data))
dev_data_loader = DataLoader(dev_dataset,
                                batch_size=128, shuffle=True)


model = tagger(len(vocab), 50, 128, len(vocab_labels), 0.3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
nepochs = 5

train_losses, dev_losses = train(model, optimizer, criterion, nepochs, train_data_loader, dev_data_loader)
# plot_results(train_losses, dev_losses, [],[])
