import sys

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from itertools import product

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
    def __init__(self, vocab_size, embedding_dim, hidden_layer, target_size, dropout_p, pre_trained_embeddings=None,
                 WINDOW_SIZE=5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * WINDOW_SIZE, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, target_size)
        self.activation = nn.Tanh()
        self.dropout1 = nn.Dropout(p=dropout_p)
        # part 3 pretrained vocab
        self.WINDOW_SIZE = WINDOW_SIZE
        # init weights
        torch.manual_seed(42)
        if pre_trained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pre_trained_embeddings))
        else:
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
            if (p == o):
                if (o == vocab_labels.stoi['O']):
                    total -= 1
                    continue
                else:
                    corrects += 1
    else:
        corrects = (pred == tags).sum()

    return corrects, total


def epoch_train(model, optimizer, criterion, train_loader, val_loader, mission='NER'):
    running_loss = 0
    running_corrects, running_total = 0, 0
    for windows, tags in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(windows)
        loss = criterion(output, tags)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(output, 1)
        batch_corrects, batch_total = accuracy(pred, tags, mission)
        running_corrects += batch_corrects
        running_total += batch_total

    return running_loss / len(train_loader), running_corrects / running_total


def evaluate(model, criterion, val_loader, mission='NER'):
    running_loss = 0
    running_corrects, running_total = 0, 0
    model.eval()
    with torch.no_grad():
        for windows, tags in (val_loader):
            output = model(windows)
            loss = criterion(output, tags)
            running_loss += loss.item()

            pred = torch.argmax(output, 1)
            batch_corrects, batch_total = accuracy(pred, tags, mission)
            running_corrects += batch_corrects
            running_total += batch_total

        return running_loss / len(val_loader), running_corrects / running_total


def train(model, optimizer, criterion, nepochs, train_loader, val_loader, mission='NER'):
    model.train()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for e in range(nepochs):
        loss, acc = epoch_train(model, optimizer, criterion, train_loader, val_loader, mission=mission)
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


def parameters_search(params_dict, train_dataset, dev_dataset, optimize='accuracy', mission='NER'):
    best_config_val_loss = 9999999
    best_config_val_accuracy = -9999999
    search_space = [dict(zip(params_dict.keys(), values)) for values in product(*params_dict.values())]
    for i, config in enumerate(search_space):
        print(f'configuration {i+1} from {len(search_space)}')
        print(f'parameters: {config}')
        train_data_loader = DataLoader(train_dataset,
                                batch_size=config['batch_size'], shuffle=True)
        dev_data_loader = DataLoader(dev_dataset,
                                batch_size=config['batch_size'], shuffle=True)
        if mission == 'NER':
            model = tagger(len(vocab), 50, config['hidden_layer'], len(vocab_labels), config['dropout_p'])
        else:
            model = tagger(len(vocab_pos), 50, config['hidden_layer'], len(vocab_labels_pos), config['dropout_p'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        nepochs = 5
        train_losses, val_losses, train_accuracy, val_accuracy = train(model, optimizer, criterion, nepochs, train_data_loader, dev_data_loader)
        if optimize=='accuracy':
            best_acc_eopch = np.argmax(val_accuracy)
            if val_accuracy[best_acc_eopch] > best_config_val_accuracy:
                best_config_val_accuracy = val_accuracy[best_acc_eopch]
                best_config = config
                best_config['nepochs'] = best_acc_eopch+1
        else: # optimize=='loss'
            best_loss_eopch = np.argmin(val_losses)
            if val_losses[best_loss_eopch] < best_config_val_loss:
                best_config_val_loss = val_losses[best_loss_eopch]
                best_config = config
                best_config['nepochs'] = best_loss_eopch+1
    return best_config

print("___________________________________NER__________________________________________________")
use_pre_trained = len(sys.argv) > 1
pre_embedding = None
if use_pre_trained:
    vocab, pre_embedding = use_pretrained('vocab.txt', 'wordVectors.txt')
    train_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, train_data))
else:
    train_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, train_data))

train_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, train_data))
dev_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, dev_data))

params_dict = {
    'hidden_layer': [170,90],
    'dropout_p': [0.4, 0.2],
    'batch_size': [128, 64],
    'lr': [1e-3, 2e-3]
    }

params_dict = { # for debuging
    'hidden_layer': [170],
    'dropout_p': [0.4],
    'batch_size': [2048],
    'lr': [1e-3]
    }

print('searching parameters...')
best_params = parameters_search(params_dict, train_dataset, dev_dataset)
print(f'Best parameters: {best_params}')

model = tagger(len(vocab), 50, best_params['hidden_layer'], len(vocab_labels), best_params['dropout_p'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), best_params['lr'])
nepochs = best_params['nepochs']
train_data_loader = DataLoader(train_dataset,
                                batch_size=best_params['batch_size'], shuffle=True)
dev_data_loader = DataLoader(dev_dataset,
                                batch_size=best_params['batch_size'], shuffle=True)
train_losses, val_losses, train_accuracy, val_accuracy = train(model, optimizer, criterion, 5, train_data_loader, dev_data_loader)
plot_results(train_losses, val_losses, train_accuracy, val_accuracy, main_title='NER')

print("___________________________________POS__________________________________________________")
if use_pre_trained:
    vocab, pre_embedding = use_pretrained('vocab.txt', 'wordVectors.txt')
    train_dataset_pos = Tagging_Dataset(data_to_window(vocab_pos, vocab_labels_pos, train_data_pos))
else:
    train_dataset_pos = Tagging_Dataset(data_to_window(vocab_pos, vocab_labels_pos, train_data_pos))

train_dataset_pos = Tagging_Dataset(data_to_window(vocab_pos, vocab_labels_pos, train_data_pos))
dev_dataset_pos = Tagging_Dataset(data_to_window(vocab_pos, vocab_labels_pos, dev_data_pos))

pos_params_dict = {
    'hidden_layer': [170,90],
    'dropout_p': [0.4, 0.2],
    'batch_size': [128, 64],
    'lr': [1e-3, 2e-3]
    }

pos_params_dict = { # for debuging
    'hidden_layer': [170],
    'dropout_p': [0.4],
    'batch_size': [4096],
    'lr': [1e-3]
    }

print('searching parameters...')
best_pos_params = parameters_search(pos_params_dict, train_dataset_pos, dev_dataset_pos, mission='POS')
print(f'Best parameters: {best_pos_params}')

model_pos = tagger(len(vocab_pos), 50, best_pos_params['hidden_layer'], len(vocab_labels_pos), best_pos_params['dropout_p'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_pos.parameters(), best_pos_params['lr'])
nepochs = best_pos_params['nepochs']
train_data_loader_pos = DataLoader(train_dataset_pos,
                                batch_size=best_pos_params['batch_size'], shuffle=True)
dev_data_loader_pos = DataLoader(dev_dataset_pos,
                                batch_size=best_pos_params['batch_size'], shuffle=True)

train_losses, val_losses, train_accuracy, val_accuracy = train(model_pos, optimizer, criterion, 5, train_data_loader_pos, dev_data_loader_pos)
plot_results(train_losses, val_losses, train_accuracy, val_accuracy, main_title='POS')
