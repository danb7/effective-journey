import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import Counter

from torch.utils.data import DataLoader

from utils import TRAIN, TEST, F2I, vocab, DEV, L2I


def feats_to_vec(features):
    F2I_fit = F2I
    vocab_fit = vocab
    V = np.zeros(len(vocab_fit))
    c = Counter()
    c.update(features)
    d = {k: v for k, v in c.items() if k in vocab_fit}
    for k in d:
        V[F2I_fit[k]] = d[k]
    # Should return a numpy vector of features.
    return V


class loglinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(600, 6)


    def forward(self, x):
        x = self.fc0(x)
        return F.log_softmax(x)

# 6.1. Train the model. (Fill empty code blocks)
def train_model(model, optimizer, criterion,
                nepochs, train_loader, val_loader, is_image_input=False):
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
        for labels, features in train_loader:
            x = feats_to_vec(features)  # convert features to a vector.
            x = torch.Tensor(x)
            y = labels if isinstance(labels, (int, float)) else L2I[labels]  # convert the label to number if needed.


            # Training pass
            model.train()  # set model in train mode
            optimizer.zero_grad()
            output = model(x)
            y_true = np.zeros(output.shape[0])
            y_true[y] = 1
            y = torch.Tensor(y_true)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # else:
            val_loss = 0
            correct = 0
            # 6.2 Evalaute model on validation at the end of each epoch.

        with torch.no_grad():
            for labels, features in val_loader:
                model.eval()
                x = feats_to_vec(features)  # convert features to a vector.
                x = torch.Tensor(x)
                y = labels if isinstance(labels, (int, float)) else L2I[
                    labels]  # convert the label to number if needed.
                y_true = np.zeros(output.shape[0])
                y_true[y] = 1
                y = torch.Tensor(y_true)

                output = model(x)
                val_loss = criterion(output, y)
                running_val_loss += val_loss.item()
                pred = output.argmax()

                correct += pred.eq(y).cpu().sum()

        # 7. track train loss and validation loss
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(running_val_loss / len(val_loader))
        # print('\nTest set: Average loss: {:.4f}, Accuracy: {}:{} ({:.0f}%)\n'.format(val_loss,correct, len(val_loader.dataset),100. *correct / len(val_loader.dataset)))
        print("Epoch: {}/{}.. ".format(e + 1, nepochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)))
    return train_losses, val_losses



epochs = 20
learning_rate = 0.01
model = loglinear()
# print(model)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
TRAIN_LOADER = DataLoader(TRAIN, batch_size=32, shuffle=True)
DEV_LOADER = DataLoader(DEV, batch_size=32, shuffle=True)

optimizer = optim.SGD(model.parameters(),lr = learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_losses, val_losses = train_model(model, optimizer, criterion, epochs,
                                       TRAIN, DEV, is_image_input=True)

