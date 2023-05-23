STUDENT={'name': 'Peleg shefi_Daniel bazar',
         'ID': '316523638_314708181'}

import numpy as np
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, vocab, n_filters, embed_dim,char_longest, WINDOW_SIZE=5):
        super().__init__()
        self.embbeding = nn.Embedding(len(vocab) + 1, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embbeding.weight)

        self.conv = nn.Conv1d(in_channels=char_longest, out_channels=n_filters, kernel_size=WINDOW_SIZE, padding=WINDOW_SIZE // 2)
        nn.init.kaiming_uniform_(self.conv.weight) 
        nn.init.constant_(self.conv.bias, 0)
        
        # self.maxpool = nn.MaxPool1d(kernel_size=30,stride=30) TODO: figure otthe numbers
        self.drop = nn.Dropout(p=0.5)

    def forward(self, char):
        embbeding = self.embbeding(char)
        x = self.drop(embbeding)
        x = self.conv(x)
        # x = self.maxpool(x).squeeze()
        # like a maxpool layer but simple
        x, _ = torch.max(x, dim=1)

        return x