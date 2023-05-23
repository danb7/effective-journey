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
        self.stoi = {k: j for j, k in self.itos.items()}

        self.freq_threshold = 0

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [tok.strip() for tok in text.split(' ')]

    def build_vocabulary(self, data):
        frequencies = {} 
        idx = 0 if self.is_labels else 1  # index from which we want our dict to start. We already used 1 indexes for unk

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

        frequencies = {k: v for k, v in frequencies.items() if v > self.freq_threshold}

        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            elif check_if_a_number(token, self.stoi.keys()) in self.stoi.keys():
                format_digits = check_if_a_number(token, self.stoi.keys())
                numericalized_text.append(self.stoi[format_digits])
            else:
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
    if all(char.isdigit() or char == '.' or char == '+' or char == '-' for char in word):
        digits_str = ""
        for char in word:
            digits_str += 'DG' if char.isdigit() else char
        digits_str = digits_str if digits_str in vocab else 'NNNUMMM'
        return digits_str
    elif all(char.isdigit() or char == ',' for char in word) and any(char.isdigit() for char in word):
        return "NNNUMMM"
    return None

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_vectors(embedding, target_vector, k):
    # Compute cosine similarities between the target vector and all other vectors in the embedding
    similarities = cosine_similarity(embedding, target_vector.reshape(1, -1))
    # Sort the cosine similarities in descending order
    sorted_indices = np.argsort(similarities, axis=0)[::-1]
    # Select the top k vectors based on the highest cosine similarities
    top_k_vectors = embedding[sorted_indices[:k].flatten()]
    return top_k_vectors


def analyze_filters(cnn_model, top_k=None):
    if top_k:
        vecs = cnn_model.cnn.embbeding.weight.data
        filter_weights = cnn_model.cnn.conv.weight.data.numpy()
        num_filters = filter_weights.shape[0]
        for i in range(num_filters):
            compare_vecs = 3 
            top_k_vecs = get_top_k_vectors(vecs, filter_weights[i],top_k)
    else:
        filter_weights = cnn_model.conv.weight.data.numpy()
        num_filters = filter_weights.shape[0]
        fig, axs = plt.subplots(nrows=num_filters, ncols=1, figsize=(5, num_filters*2))

        for i in range(num_filters):
            axs[i].imshow(filter_weights[i, 0, :, :], cmap='gray')
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()