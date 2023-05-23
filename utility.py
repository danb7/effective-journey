import math
import numpy as np
from matplotlib import pyplot as plt

SOS = '<s>'
EOS = '</s>'
UNK = 'UUUNKKK'
def read_data(fname, splitter,lower=False):
    sentence_list = []
    with open(fname, encoding="utf8") as f:
        lines = f.readlines()
    sentence = ""
    labels = ""
    for line in lines:
        if line != '\n':
            text, label = line.strip().split(splitter, 1)  # TODO: check lower
            text = text.lower() if lower else text
            sentence = sentence + text + " "
            labels = labels + label + " "
        else:
            sentence = sentence[:-1]
            labels = labels[:-1]
            sentence_list.append((f'{SOS} {SOS} {sentence} {EOS} {EOS}', labels))
            sentence = ""
            labels = ""
    return sentence_list

def read_test_file(fname):
    sentence_list = []
    with open(fname, encoding="utf8") as f:
        lines = f.readlines()
    sentence = ""
    for line in lines:
        if line != '\n':
            text = line.strip()
            sentence = sentence + text + " "
        else:
            sentence = sentence[:-1]
            sentence_list.append(f'{SOS} {SOS} {sentence} {EOS} {EOS}')
            sentence = ""

    return sentence_list


def use_pretrained(vocab, embeddings):
    with open(vocab, encoding="utf8") as f:
        lines = f.readlines()
    words = []
    for line in lines:
        words.append(line.strip())
    pre_vocab = Vocabulary()
    pre_vocab.build_vocabulary(words[1:])
    vecs = np.loadtxt(embeddings)
    return pre_vocab, vecs


class Vocabulary:
    def __init__(self, is_labels=False, sub_word=None):
        '''
        sub_word : str, default None
            [None | prefix | suffix]
        '''
        self.is_labels = is_labels
        self.sub_word = sub_word
        self.itos = {}
        if not is_labels:
            self.itos = {0: UNK}
            if sub_word=='prefix':
                self.itos = {0: UNK[:3]}
            elif sub_word=='suffix':
                self.itos = {0: UNK[-3:]}
        # initiate the token to index dict
        self.stoi = {k: j for j, k in self.itos.items()}

        self.freq_threshold = 0

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):  # split on space and converts the sentence to list of words
        return [tok.strip() for tok in text.split(' ')]

    def build_vocabulary(self, data):
        frequencies = {}  # init the freq dict
        idx = 0 if self.is_labels else 1  # index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk

        # calculate freq of words
        for sentence in data:
            for word in self.tokenizer(sentence):
                prefix = word[:3]
                suffix = word[-3:]
                if self.sub_word=='prefix':
                    word = prefix
                elif self.sub_word=='suffix':
                    word = suffix
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        # limit vocab by removing low freq words
        frequencies = {k: v for k, v in frequencies.items() if v > self.freq_threshold}

        # create vocab
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    def numericalize(self, text):
        # tokenize text
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            elif check_if_a_number(token, self.stoi.keys()) in self.stoi.keys():
                format_digits = check_if_a_number(token, self.stoi.keys())
                numericalized_text.append(self.stoi[format_digits])
            else:  # out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi[UNK])

        return numericalized_text

    def to_text(self, numericalized_text):
        return self.itos[numericalized_text.item()] if self.is_labels else self.itos[numericalized_text[2].item()]


def sentence_to_windows(vocab, vocab_labels, sentence, labels=None, window_size=5):
    numerized_senetnce = vocab.numericalize(sentence)
    windows_sentence = [(numerized_senetnce[i - 2], numerized_senetnce[i - 1], numerized_senetnce[i],
                         numerized_senetnce[i + 1], numerized_senetnce[i + 2]) for i in
                        range(2, len(numerized_senetnce) - 2)]
    if labels:
        numerized_label = vocab_labels.numericalize(labels)
        return windows_sentence, numerized_label
    else:
        return windows_sentence


def data_to_window(vocab, vocab_labels, data, include_labels=True, window_size=5):
    '''Transform data sentences to windows per word

    Parameters
    ----------
    include_labels : bool, default True
        Whether to return labels in addition to sentences.
        Useful when transforming test data
    '''
    windows_sentences = []
    windows_labels = []
    for row in data:
        if include_labels:
            sentence, tags = row
            windows, numerized_tags = sentence_to_windows(vocab, vocab_labels, sentence, tags)
        else:
            windows = sentence_to_windows(vocab, vocab_labels, row)
        windows_sentences.extend(windows)
        if include_labels:
            windows_labels.extend(numerized_tags)
    if include_labels:
        return windows_sentences, windows_labels
    else:
        return windows_sentences

def plot_results(train_loss, val_loss, train_acc, val_acc, main_title=''):
    """
    This function takes lists of values and creates side-by-side graphs to show training and validation performance
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(
        train_loss, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[0].plot(
        val_loss, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].set_xticks(range(0,len(train_loss)))
    ax[1].plot(
        train_acc, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[1].plot(
        val_acc, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].set_xticks(range(0,len(train_loss)))
    fig.suptitle(main_title)
    # plt.show()
    best_acc=round(max(val_acc), 2)
    best_loss=round(min(val_loss), 2)
    fig.savefig(f'graphs/{main_title}_plot_{best_acc}_acc_{best_loss}_loss.png')


def create_vocabs(train_data):
    sentences, labels = zip(*train_data)
    vocab = Vocabulary()
    vocab_labels = Vocabulary(is_labels=True)
    vocab.build_vocabulary(sentences)
    vocab_labels.build_vocabulary(labels)
    return vocab, vocab_labels

def create_pre_suf_vocabs(vocab):
    pre_vocab = Vocabulary(sub_word='prefix')
    suf_vocab = Vocabulary(sub_word='suffix')
    pre_vocab.build_vocabulary(vocab.stoi.keys())
    suf_vocab.build_vocabulary(vocab.stoi.keys())
    return pre_vocab, suf_vocab

def create_cnn_vocabs(vocab):
    characters = list(set(list((''.join(vocab.stoi.keys())))))
    cnn_vocab = Vocabulary()
    cnn_vocab.build_vocabulary(characters)
    return cnn_vocab

def check_if_a_number(word, vocab):

    # If a number is of pattern 'DGDG', 'DG.DG', '.DG', '+DG', '-DG' and etc.
    if all(ch.isdigit() or ch == '.' or ch == '+' or ch == '-' for ch in word):
        pattern = ""

        # Replace each character with 'DG'
        for ch in word:
            pattern += 'DG' if ch.isdigit() else ch

        # If this pattern is in the pre-trained vocabulary return it; Otherwise, return the pattern 'NNNUMMM'
        pattern = pattern if pattern in vocab else 'NNNUMMM'
        return pattern

    # If a number is of pattern '_ ,_ _' ; '_ ,_ _ _, _ _ _' and etc return the pattern 'NNNUMMM'.
    elif all(ch.isdigit() or ch == ',' for ch in word) and any(ch.isdigit() for ch in word):
        return "NNNUMMM"

    return None


def get_conv2d_layer_shape(layer, in_shape):
    """Return the shape of torch.nn.Conv2d layer.

    Parameters
    ----------
    layer: The Conv2d layer.
    in_shape: The input shape.

    returns
    -------
    The shape of the Conv2d layer
    """
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


def get_maxpool2d_layer_shape(layer, in_shape):
    """
    @brief: Return the shape of torch.nn.MaxPool2d layer.

    @param layer: The MaxPool2d layer.
    @param in_shape: The input shape.

    @return: The shape of the MaxPool2d layer.
    """
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

# def most_similiar(word, k):
#     query_vector = words[word]
#     similarities = np.zeros(vecs.shape[0])
#     for i, vector in enumerate(vecs):
#         similarities[i] = cosine_similarity(query_vector, vector)
#     indices = np.argsort(similarities)[::-1][1:k + 1]
#     top_k_similar_vectors = vecs[indices]
#     return top_k_similar_vectors, indices, similarities[indices]


def analyze_filters(cnn_model, analyze_type='top_k'):
    if analyze_type=='top_k':
        vecs = cnn_model.cnn.embbeding.weight.data
        filter_weights = cnn_model.cnn.conv.weight.data.numpy()
        num_filters = filter_weights.shape[0]
        for i in range(num_filters):
            pass
    else:
        filter_weights = cnn_model.conv.weight.data.numpy()
        num_filters = filter_weights.shape[0]
        fig, axs = plt.subplots(nrows=num_filters, ncols=1, figsize=(5, num_filters*2))

        for i in range(num_filters):
            axs[i].imshow(filter_weights[i, 0, :, :], cmap='gray')
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()