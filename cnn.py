import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_filters, filter_sizes, pool_size, dropout,
                 char_padding=2):
        super().__init__()
        self.embbeding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_size))
             for fs in filter_sizes])
        self.max_pool1 = nn.MaxPool1d(pool_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(12 * n_filters, 1, bias=True) #TODO: need to understand the shape here.
        print(n_filters)
    def forward(self, text, text_length):
        embedded = self.embbeding(text)
        embedded = embedded.unsqueeze(1)
        convolution = [conv(embedded) for conv in self.convs]
        max1 = self.max_pool1(convolution[0].squeeze())
        max2 = self.max_pool1(convolution[0].squeeze())
        cat = torch.cat((max1, max2), dim=2)
        x = cat.view(cat.shape[0], -1)
        x = self.fc1(self.relu(x))
        x = self.dropout(x)
        return x
