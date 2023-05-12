#
from collections import Counter

import torch

EOS = 'end of sentence'
SOS = 'start of sentence'


def read_data(fname):
    data = []
    dict = {'per': 0, 'loc': 0, 'org': 0, 'time': 0, 'o': 0, 'misc': 0}
    with open(fname, encoding="utf8") as f:
        lines = f.readlines()
    sentence = [SOS, SOS]
    labels = ['o', 'o']
    for line in lines:
        if line != '\n':
            text, label = line.strip().lower().split("\t", 1)  # TODO: check lower
            dict[label] = dict[label] + 1
            sentence.append(text)
            labels.append(label)
        else:
            sentence.extend([EOS, EOS])  # TODO: check later
            labels.extend(['o', 'o'])
            data.append((sentence, labels))
            sentence = [SOS, SOS]
            labels = ['o', 'o']
    return data


TRAIN = read_data("ner/train")
DEV = read_data("ner/dev")

# NER Labels to IDs
# NER2ID = {'per': 0, 'loc': 1, 'org': 2, 'time': 3, 'o': 4, 'misc': 5}
labels = sorted(set([label for f, l in TRAIN for label in l]))
LABEL2ID = dict(zip(labels, range(len(labels))))

vocab = sorted(set([word for f, l in TRAIN for word in f]))
vocab.insert(0, 'NOT SEEN')  # words that are not in the train set
FEAT2ID = dict(zip(vocab, range(len(vocab))))


def prepare_sequence(seq, to_ix):
    """Input: takes in a list of words, and a dictionary containing the index of the words
    Output: a tensor containing the indexes of the word"""
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def create_ids(training_data):
    word_to_ix = {}  # This is the word dictionary which will contain the index to each word
    for sent in training_data:
        for word in sent:
            if word not in word_to_ix.keys():
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix


