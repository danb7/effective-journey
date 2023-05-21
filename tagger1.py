import sys

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from itertools import product
import argparse

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
    def __init__(self, vocab, embedding_dim, hidden_layer, target_size, dropout_p, 
                 pre_trained_embeddings=None, prefix_vocab=None, suffix_vocab=None, 
                 cnn_vocab=None, char_embedding_dim=30, cnn_window_size=3, cnn_padding_size=2, 
                 dropout_p_cnn=0.5, WINDOW_SIZE=5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.prefix_vocab = prefix_vocab
        self.suffix_vocab = suffix_vocab
        self.cnn_vocab = cnn_vocab
        self.word_embeddings = nn.Embedding(len(self.vocab), embedding_dim)
        if prefix_vocab:
            self.prefix_embeddings = nn.Embedding(len(self.prefix_vocab), embedding_dim)
            self.suffix_embeddings = nn.Embedding(len(self.suffix_vocab), embedding_dim)
        elif cnn_vocab:
            self.char_embeddings = nn.Embedding(len(self.cnn_vocab), char_embedding_dim, padding_idx=0)
            self.cnn_layer = nn.Conv2d(in_channels=1, out_channels=max(self.cnn_vocab.stoi.keys(), key=len), 
                                       kernel_size=(cnn_window_size,char_embedding_dim), 
                                       padding=(cnn_padding_size,0))
            conv2d_layer_shape = get_conv2d_layer_shape(self.cnn_layer, (max(self.cnn_vocab.stoi.keys(), key=len), char_embedding_dim))
            self.char_maxpool_layer = torch.nn.MaxPool2d(kernel_size=(conv2d_layer_shape[0], 1))
            self.dropout_cnn_layer = torch.nn.Dropout(p=dropout_p_cnn)
            self.fc_input_dim = (self.WINDOW_SIZE *
                                 (self.embedding_dim +
                                  self.cnn_char_num_filters *
                                  self.maxpool2d_layer_shape[2] *
                                  self.maxpool2d_layer_shape[3]))
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
        if prefix_vocab:
            nn.init.xavier_uniform_(self.prefix_embeddings.weight)
            nn.init.xavier_uniform_(self.suffix_embeddings.weight)
        elif cnn_vocab:
            nn.init.kaiming_uniform_(self.char_embeddings) # check in/out
        nn.init.xavier_uniform_(self.fc1.weight, gain=torch.nn.init.calculate_gain('tanh'))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, windows):
        if self.prefix_vocab:
            prefix_window, suffix_window = self._get_subword_prefix_suffix(windows)
            prefix_emb = self.prefix_embeddings(prefix_window).view(-1, self.embedding_dim * self.WINDOW_SIZE)
            word_emb = self.word_embeddings(windows).view(-1, self.embedding_dim * self.WINDOW_SIZE)
            suffix_emb = self.suffix_embeddings(suffix_window).view(-1, self.embedding_dim * self.WINDOW_SIZE)
            x = prefix_emb + word_emb + suffix_emb
        elif self.cnn_vocab:
            x = self.char_embeddings(windows).view(-1, self.embedding_dim * self.WINDOW_SIZE)
        else:
            x = self.word_embeddings(windows).view(-1, self.embedding_dim * self.WINDOW_SIZE)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=-1)
        return x
    
    def _get_subword_prefix_suffix(self, windows):
        prefix_list = []
        suffix_list = []
        iterate_window = [windows] if windows.dim() == 1 else windows
        for window in iterate_window:
            prefix = [self.prefix_vocab.stoi[self.vocab.itos[w.item()][:3]] for w in window]
            suffix = [self.suffix_vocab.stoi[self.vocab.itos[w.item()][-3:]] for w in window]
            prefix_list.append(prefix)
            suffix_list.append(suffix)
        return torch.LongTensor(prefix_list), torch.LongTensor(suffix_list)


def accuracy(pred, tags, mission):
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
        # corrects = (pred == tags).sum()
        for p, o in zip(pred, tags):
            if (p == o):
                corrects += 1
    return corrects, total


def epoch_train(model, optimizer, criterion, train_loader, val_loader, mission, grad_clip=0):
    running_loss = 0
    running_corrects, running_total = 0, 0
    model.train()
    for windows, tags in tqdm(train_loader):
        optimizer.zero_grad()
        output = model(windows)
        loss = criterion(output, tags)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        pred = torch.argmax(output, 1)
        batch_corrects, batch_total = accuracy(pred, tags, mission)
        running_corrects += batch_corrects
        running_total += batch_total

    return running_loss / len(train_loader), running_corrects / running_total


def evaluate(model, criterion, val_loader, mission):
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


def test_prediction(model, test_data):
    '''Predict on test data.

    Parameters
    ----------
    model : tagger
    test_data : list
        list of windows

    Returns
    -------
    predictions : Tensor
        predictions for every word
    '''
    model.eval()
    predictions = []
    with torch.no_grad():
        for window in test_data:
            window = window[0]
            outputs = model(window)
            pred = torch.argmax(outputs, 1)
            predictions.append(pred)
    predictions = torch.cat(predictions)  # Concatenate predictions into a single tensor
    return predictions

def save_test_file(test_data, predictions_labels, save_path=None, seperator=' '):
    '''save test predictions to file.

    Parameters
    ----------
    save_path : str
        save the test predictions file in that path.
    seperator : str
        seperator between word and tag
    '''
    f = open(save_path,"w")
    i=0
    for sentence in test_data:
        for word in sentence.split(' '):
            f.write(f'{word}{seperator}{predictions_labels[i]}\n')
            i+=1
        f.write("\n")
        

def train(model, optimizer, criterion, nepochs, train_loader, val_loader, mission, scheduler=None, grad_clip=0, return_best_epoch=True, optimize='accuracy'):
    '''Train the model epoch by epoch.

    Parameters
    ----------
    model : tagger
    optimizer : optimizer
    criterion : Loss function
    n_eopchs : int
        number of epoch to train the model
    train_loader : Tagging_Dataset
        the train data
    val_loader : Tagging_Dataset
        the validation data
    mission : ['NER' | 'POS']
    return_best_epoch : bool, default True
        weather to return in addition the best model at the best epoch
    optimize : [accuracy | loss], default "accuracy"
        Return the best configuration based on the specified [optimize]

    Returns
    -------
    return 4 list of train and validation loss and accuracy per epoch
    optional: if return_best_epoch is True return also the best model
    '''
    model.train()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = 9999999
    best_val_accuracy = -9999999
    for e in range(nepochs):
        loss, acc = epoch_train(model, optimizer, criterion, train_loader, val_loader, mission, grad_clip)
        val_loss, val_acc = evaluate(model, criterion, val_loader, mission)
        train_losses += [loss]
        val_losses += [val_loss]
        train_accs += [acc]
        val_accs += [val_acc]

        if return_best_epoch:
            if optimize=='accuracy':
                if val_acc > best_val_accuracy:
                    best_val_accuracy = val_acc
                    best_model = model
            else: # optimize=='loss'
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
        if scheduler:
            scheduler.step()

        print("Epoch: {}/{}.. ".format(e + 1, nepochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Validation Loss: {:.3f}.. ".format(val_losses[-1]),
              f'Accuracy train: {(100 * (train_accs[-1]))}',
              f'Accuracy validation: {(100 * (val_accs[-1]))}')
    if return_best_epoch:
        return train_losses, val_losses, train_accs, val_accs, best_model
    else:
        return train_losses, val_losses, train_accs, val_accs


def parameters_search(params_dict, n_eopchs, train_dataset, dev_dataset, vocab, return_best_epoch=True, optimize='accuracy', mission='NER', pre_trained_emb=None, pre_vocab=None, suf_vocab=None, cnn_vocab=None):
    '''Exhaustive search over specified parameter values

    Parameters
    ----------
    params_dict : dict
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values.
    n_eopchs : int
        number of epoch to run per configuration
    return_best_epoch : bool, default True
        weather to return in addition the best model at the best epoch
    optimize : [accuracy | loss], default "accuracy"
        Return the best configuration based on the specified [optimize]
    pre_trained_emb : Embedding matrix, default None
        if not none, use that embedding matrix

    Returns
    -------
    dict : {
        'best_config': dict
            best parameters
        'model': tagger
            [best model | model after all epoches] based on return_best_epoch 
        'train_losses': list
            train losses for all epochs
        'val_losses': list
            val losses for all epochs
        'train_accuracy': list
            train accuracy for all epochs
        'val_accuracy': list
            val accuracy for all epochs
    }

    Notes
    -----
    Currently returning the model after all epochs and not the one after the n best epoch
    '''
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
            model = tagger(vocab, 50, config['hidden_layer'], len(vocab_labels), config['dropout_p'], pre_trained_emb, prefix_vocab=pre_vocab, suffix_vocab=suf_vocab, cnn_vocab=cnn_vocab)
        else:
            model = tagger(vocab_pos, 50, config['hidden_layer'], len(vocab_labels_pos), config['dropout_p'], pre_trained_emb, prefix_vocab=pre_vocab, suffix_vocab=suf_vocab, cnn_vocab=cnn_vocab)
        criterion = nn.CrossEntropyLoss()
        if cnn_vocab:
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
            initial_learning_rate = 0.015 if mission=='NER' else 0.01
            lambda_lr = lambda epoch: initial_learning_rate/(1+0.05 *epoch)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
            grad_clip = 5
        else:
            lr_scheduler = None
            grad_clip = 0
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        train_losses, val_losses, train_accuracy, val_accuracy, best_config_model = train(model, optimizer, criterion, n_eopchs, train_data_loader, dev_data_loader, mission, lr_scheduler, grad_clip=grad_clip)
        if optimize=='accuracy':
            best_acc_eopch = np.argmax(val_accuracy)
            if val_accuracy[best_acc_eopch] > best_config_val_accuracy:
                best_config_val_accuracy = val_accuracy[best_acc_eopch]
                best_config = config
                best_config['nepochs'] = best_acc_eopch+1
                best_model = best_config_model if return_best_epoch else model
                best_train_losses, best_val_losses, best_train_accuracy, best_val_accuracy = train_losses, val_losses, train_accuracy, val_accuracy
        else: # optimize=='loss'
            best_loss_eopch = np.argmin(val_losses)
            if val_losses[best_loss_eopch] < best_config_val_loss:
                best_config_val_loss = val_losses[best_loss_eopch]
                best_config = config
                best_config['nepochs'] = best_loss_eopch+1
                best_model = best_config_model if return_best_epoch else model
                best_train_losses, best_val_losses, best_train_accuracy, best_val_accuracy = train_losses, val_losses, train_accuracy, val_accuracy
    
    return {
        'best_config': best_config,
        'model': best_model,
        'train_losses': best_train_losses, 
        'val_losses': best_val_losses, 
        'train_accuracy': best_train_accuracy, 
        'val_accuracy': best_val_accuracy
    }

print("___________________________________NER__________________________________________________")
parser = argparse.ArgumentParser(description="NLP tagging task")
group = parser.add_mutually_exclusive_group()
parser.add_argument("-p", "--pretrained", action="store_true")
group.add_argument("-s", "--subword", action="store_true")
group.add_argument("-c", "--cnn", action="store_true")
args = parser.parse_args()
pre_embedding = None
pre_vocab = None
suf_vocab = None
char_vocab = None

train_data = read_data('ner/train', '\t', lower=args.pretrained)
dev_data = read_data('ner/dev', '\t', lower=args.pretrained)
vocab, vocab_labels = create_vocabs(train_data)

if args.pretrained:
    print('using pre-trained embedding\n')
    vocab, pre_embedding = use_pretrained('vocab.txt', 'wordVectors.txt')

if args.subword:
    print('using subword embedding\n')
    pre_vocab, suf_vocab = create_pre_suf_vocabs(vocab)

if args.cnn:
    print('using character embedding\n')
    char_vocab = create_cnn_vocabs(vocab)


train_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, train_data))
dev_dataset = Tagging_Dataset(data_to_window(vocab, vocab_labels, dev_data))

params_dict = {
    'hidden_layer': [170, 130, 90],
    'dropout_p': [0.5, 0.4, 0.3],
    'batch_size': [256, 128, 64],
    'lr': [1e-4, 5e-5, 1e-5]
    }
# best parameters:
# {'hidden_layer': 130, 'dropout_p': 0.3, 'batch_size': 128, 'lr': 0.0001, 'nepochs': 6}
best_params_dict = {'hidden_layer': [130], 'dropout_p': [0.3], 'batch_size': [128], 'lr': [1e-4]}
print('searching parameters...\n')
best_tagger = parameters_search(best_params_dict, 8, train_dataset, dev_dataset, vocab, mission='NER', pre_trained_emb=pre_embedding, pre_vocab=pre_vocab, suf_vocab=suf_vocab, cnn_vocab=char_vocab)
plot_results(best_tagger['train_losses'], best_tagger['val_losses'],\
    best_tagger['train_accuracy'], best_tagger['val_accuracy'], main_title=f'NER_P-{args.pretrained}_S-{args.subword}')
print(f'best parameters:\n{best_tagger["best_config"]}')

# saving test predictions
test_data = read_test_file('ner/test') 
test_dataset = TensorDataset(torch.LongTensor(data_to_window(vocab, vocab_labels, test_data, include_labels=False)))
test_preds = test_prediction(best_tagger['model'], test_dataset)
test_preds_labels = [vocab_labels.itos[p.item()] for p in test_preds]
# save_test_file(test_data, test_preds_labels, 'test1.ner', seperator='\t')

print("___________________________________POS__________________________________________________")
train_data_pos = read_data('pos/train', ' ', lower=args.pretrained)
dev_data_pos = read_data('pos/dev', ' ', lower=args.pretrained)
vocab_pos, vocab_labels_pos = create_vocabs(train_data_pos)
pre_vocab_pos = None
suf_vocab_pos = None
if args.pretrained:
    print('using pre-trained embedding\n')
    vocab_pos, pre_embedding = use_pretrained('vocab.txt', 'wordVectors.txt')

if args.subword:
    print('using subword embedding\n')
    pre_vocab_pos, suf_vocab_pos = create_pre_suf_vocabs(vocab_pos)

train_dataset_pos = Tagging_Dataset(data_to_window(vocab_pos, vocab_labels_pos, train_data_pos))
dev_dataset_pos = Tagging_Dataset(data_to_window(vocab_pos, vocab_labels_pos, dev_data_pos))

pos_params_dict = { # for debuging i used only one item per and very big batch
    'hidden_layer': [170, 90],
    'dropout_p': [0.4, 0.2],
    'batch_size': [128, 64],
    'lr': [1e-4, 5e-5]
    }
# best parameters:
# {'hidden_layer': 90, 'dropout_p': 0.2, 'batch_size': 64, 'lr': 5e-05, 'nepochs': 8}
best_pos_params_dict = {'hidden_layer': [90], 'dropout_p': [0.2], 'batch_size': [64], 'lr': [5e-05]}
print('searching parameters...\n')
best_tagger_pos = parameters_search(best_pos_params_dict, 8, train_dataset_pos, dev_dataset_pos, vocab_pos, mission='POS', pre_trained_emb=pre_embedding, pre_vocab=pre_vocab_pos, suf_vocab=suf_vocab_pos)
plot_results(best_tagger_pos['train_losses'], best_tagger_pos['val_losses'],\
    best_tagger_pos['train_accuracy'], best_tagger_pos['val_accuracy'], main_title=f'POS_P-{args.pretrained}_S-{args.subword}')
print(f'best parameters:\n{best_tagger_pos["best_config"]}')

# saving test predictions
test_data_pos = read_test_file('pos/test') 
test_dataset_pos = TensorDataset(torch.LongTensor(data_to_window(vocab_pos, vocab_labels_pos, test_data_pos, include_labels=False)))
test_preds_pos = test_prediction(best_tagger_pos['model'], test_dataset_pos)
test_preds_labels_pos = [vocab_labels_pos.itos[p.item()] for p in test_preds_pos]
# save_test_file(test_data_pos, test_preds_labels_pos, 'test1.pos', seperator=' ')
