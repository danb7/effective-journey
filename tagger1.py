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
        #init weights
        torch.manual_seed(42)
        nn.init.xavier_uniform_(self.word_embeddings.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, sentence):
        x = self.word_embeddings(sentence).view(-1, self.embedding_dim * self.WINDOW_SIZE)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x
    
def accuracy(pred, tags, mission='NER'):
    corrects = 0
    total = len(pred)
    if mission.upper() == 'NER':
        for p, o in zip(pred, tags):
            if (p==o):
                if (o==vocab_labels.stoi['O']):
                    total -= 1
                    continue
                else:
                    corrects += 1
    else:
        corrects = (pred==tags).sum()

    return corrects, total


def epoch_train(model, optimizer, criterion, train_loader, val_loader, mission='NER'):
    running_loss = 0
    running_corrects, running_total = 0, 0
    for windows, tags in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(windows)
        loss = criterion(output, tags)
        running_loss += loss
        loss.backward()
        optimizer.step()

        pred = torch.argmax(output, 1)
        batch_corrects, batch_total = accuracy(pred, tags, mission)
        running_corrects += batch_corrects
        running_total += batch_total

    return running_loss/len(train_loader), running_corrects/running_total

def evaluate(model, criterion, val_loader, mission='NER'):
    running_loss = 0
    running_corrects, running_total = 0, 0
    model.eval()
    with torch.no_grad():
        for windows, tags in (val_loader):
            output = model(windows)
            loss = criterion(output, tags)
            running_loss += loss

            pred = torch.argmax(output, 1)
            batch_corrects, batch_total = accuracy(pred, tags, mission)
            running_corrects += batch_corrects
            running_total += batch_total

        return running_loss/len(val_loader), running_corrects/running_total


def train(model, optimizer, criterion, nepochs, train_loader, val_loader, mission='NER'):
    model.train()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for e in range(nepochs):
        loss, acc = epoch_train(model, optimizer, criterion, train_loader, val_loader, mission='NER')
        val_loss, val_acc = evaluate(model, criterion, val_loader)
        train_losses += [loss]
        val_losses += [val_loss]
        train_accs += [acc]
        val_accs += [val_acc]
        print("Epoch: {}/{}.. ".format(e + 1, nepochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Validation Loss: {:.3f}.. ".format(val_losses[-1]),
              f'Accuracy train: {(100 * (train_accs[-1]))}',
              f'Accuracy validation: {(100 * (val_accs[-1]))}')

    return train_losses, val_losses, train_accs, val_accs



print("___________________________________NER__________________________________________________")

train_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, train_data))
train_data_loader = DataLoader(train_dataset,
                                batch_size=128, shuffle=True)



dev_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, dev_data))
dev_data_loader = DataLoader(dev_dataset,
                                batch_size=128, shuffle=True)


model = tagger(len(vocab), 50, 128, len(vocab_labels), 0.4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
nepochs = 5

train_losses, val_losses, train_accuracy, val_accuracy = train(model, optimizer, criterion, nepochs, train_data_loader, dev_data_loader)
plot_results(train_losses, val_losses, train_accuracy, val_accuracy)

print("___________________________________POS__________________________________________________")
vocab_pos,vocab_labels_pos,train_data_pos,dev_data_pos = create_data('pos/train')


train_dataset_pos = Tagging_Dataset(data_to_window(vocab_pos, vocab_labels_pos, train_data_pos))
train_data_loader_pos = DataLoader(train_dataset_pos,
                                batch_size=128, shuffle=True)
dev_dataset_pos = Tagging_Dataset(data_to_window(vocab_pos, vocab_labels_pos, dev_data_pos))
dev_data_loader_pos = DataLoader(dev_dataset_pos,
                                batch_size=128, shuffle=True)

model_pos = tagger(len(vocab_pos), 50, 128, len(vocab_labels_pos), 0.4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pos.parameters())
nepochs = 5

train_losses, val_losses, train_accuracy, val_accuracy = train(model_pos, optimizer, criterion, nepochs, train_data_loader_pos, dev_data_loader_pos)