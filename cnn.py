import math
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_filters, filter_size, dropout,
                 char_longest=20, char_padding=2):
        super().__init__()
        self.embbeding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        nn.init.kaiming_uniform_(self.embbeding.weight)  # check in/out
        self.conv = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_size, embed_size), padding=(char_padding,0), stride=(1,1))
        self.char_longest = char_longest
        self.conv2d_layer_shape = self._get_conv2d_layer_shape((char_longest,embed_size))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(self.conv2d_layer_shape[0], 1))
        self.maxpool_layer_shape = self._get_maxpool2d_layer_shape(self.conv2d_layer_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        cnn_char_features = []
        for i in range(text.shape[1]):
            word_chars = text[:, i, :]
            word_chars = word_chars.unsqueeze(1)
            embedded = self.embbeding(word_chars)
            embedded = self.dropout(embedded)
            # embedded = embedded.unsqueeze(1)
            convolution = self.conv(embedded)
            max1 = self.max_pool1(convolution).view(text.shape[0],-1)
            cnn_char_features.append(max1)#.unsqueeze(1))
        cnn_char_features = torch.cat(cnn_char_features, dim=1)
        return cnn_char_features
    def _get_conv2d_layer_shape(self, in_shape):
        """Return the shape of torch.nn.Conv2d layer.

        Parameters
        ----------
        layer: The Conv2d layer.
        in_shape: The input shape.

        returns
        -------
        The shape of the Conv2d layer
        """
        layer = self.conv
        H_in, W_in = in_shape

        kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size
        dilation = (layer.dilation, layer.dilation) if isinstance(layer.dilation, int) else layer.dilation
        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride

        if isinstance(layer.padding, str):
            if layer.padding == "valid":
                H_out = math.floor((H_in + dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1)
                W_out = math.floor((W_in + dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1)
            elif layer.padding == "same":
                H_out = H_in
                W_out = W_in
        else:
            padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

            H_out = math.floor((H_in + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1)
            W_out = math.floor((W_in + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1)

        return (H_out, W_out)
    

    def _get_maxpool2d_layer_shape(self, in_shape):
        """
        @brief: Return the shape of torch.nn.MaxPool2d layer.

        @param layer: The MaxPool2d layer.
        @param in_shape: The input shape.

        @return: The shape of the MaxPool2d layer.
        """
        layer = self.max_pool1
        H_in, W_in = in_shape

        kernel_size = (layer.kernel_size, layer.kernel_size) if isinstance(layer.kernel_size, int) else layer.kernel_size
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

        if hasattr(layer, 'dilation'):
            dilation = (layer.dilation, layer.dilation) if isinstance(layer.dilation, int) else layer.dilation
        else:
            dilation = (1, 1)

        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride

        round_op = round_op = math.ceil if layer.ceil_mode else math.floor

        if isinstance(layer.padding, str):
            if layer.padding == "valid":
                H_out = round_op((H_in + dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1)
                W_out = round_op((W_in + dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1)
            elif layer.padding == "same":
                H_out = H_in
                W_out = W_in
        else:
            padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding

            H_out = round_op((H_in + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1)
            W_out = round_op((W_in + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1)

        return (H_out, W_out)