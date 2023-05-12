import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utility import TRAIN, DEV, create_ids, LABEL2ID, FEAT2ID, EOS, SOS

TRAIN_LOADER = DataLoader(TRAIN, batch_size=1, shuffle=False)
DEV_LOADER = DataLoader(DEV, batch_size=1, shuffle=True)


# # get vocabulary
# V = create_vocabulary(TRAIN_LOADER)
# vocab_ids = create_ids(V)


class tagger(nn.Module):
    def __init__(self, embedding_dim, hidden_layer, vocab_size, target_size, WINDOW_SIZE):
        super().__init__()
        self.WINDOW_SIZE = WINDOW_SIZE
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * WINDOW_SIZE, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, target_size)

    def forward(self, sentence):
        x = self.word_embeddings(sentence)
        x = x.flatten()
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x)


def train_model(model, optimizer, criterion,
                nepochs, train_loader, val_loader):
    '''
    Train a pytorch model and evaluate it every epoch.
    Params:
    model - a pytorch model to train
    optimizer - an optimizer
    criterion - the criterion (loss function)
    nepochs - number of training epochs
    train_loader - dataloader for the trainset
    val_loader - dataloader for the valset
    is_image_input (default False) - If false, flatten 2d images into a 1d array.
                                  Should be True for Neural Networks
                                  but False for Convolutional Neural Networks.
    '''
    train_losses, val_losses = [], []

    for e in range(nepochs):
        running_loss = 0
        running_val_loss = 0
        model.train()  # set model in train mode
        for sentence, labels in train_loader:
            # Training pass
            optimizer.zero_grad()
            for i_word, word in enumerate(sentence):
                if (word == SOS or word == EOS):
                    continue

                # x = [sentence[i_word - 2], sentence[i_word - 1], sentence[i_word], sentence[i_word + 1],
                #      sentence[i_word + 2]]
                # x = [sentence[i_word - np.floor((model.WINDOW_SIZE / 2)) + i] for i in range(model.WINDOW_SIZE)]
                x = [sentence[i_word + i] for i in
                     range(-1 * int((model.WINDOW_SIZE / 2)), int((model.WINDOW_SIZE / 2))+1)]  # build window
                x = [FEAT2ID.get(word, 0) for word in x]  # change string to id,return 0 if not found
                x = torch.tensor(x)
                output = model(x)
                target = torch.zeros(len(output))
                target[LABEL2ID[labels[i_word]]] = 1
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print(loss)
                # pred = np.argmax(output)


        correct = 0
        with torch.no_grad():
            for words, label in val_loader:
                model.eval()
                for i_word, word in enumerate(sentence):
                    if (word == SOS or word == EOS):
                        continue
                    x = [sentence[i_word + i] for i in
                         range(-1 * np.floor((model.WINDOW_SIZE / 2)),
                               np.ceil((model.WINDOW_SIZE / 2)))]  # build window
                    x = [FEAT2ID.get(word, 0) for word in x]  # change string to id,return 0 if not found
                    output = model(x)
                    val_loss = criterion(output, label)
                    running_val_loss += val_loss.item()
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(label.view_as(pred)).cpu().sum()

        # 7. track train loss and validation loss
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(running_val_loss / len(val_loader))
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}:{} ({:.0f}%)\n'.format(val_loss,correct, len(val_loader.dataset),100. *correct / len(val_loader.dataset)))
        print("Epoch: {}/{}.. ".format(e + 1, nepochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)))
    return train_losses, val_losses


model = tagger(50, 5, len(FEAT2ID), len(LABEL2ID), 5)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
nepochs = 5

train_losses, val_losses = train_model(model, optimizer, loss, nepochs,
                                       TRAIN, DEV)
# Define the embedding layer with 10 vocab size and 50 vector embeddings.
# print(nn.init.xavier_uniform_(embedding.weight))
