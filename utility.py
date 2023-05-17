from matplotlib import pyplot as plt


def read_data(fname, splitter):
    sentence_list = []
    with open(fname, encoding="utf8") as f:
        lines = f.readlines()
    sentence = ""
    labels = ""
    for line in lines:
        if line != '\n':
            text, label = line.strip().split(splitter, 1)  # TODO: check lower
            # text = text.lower()
            sentence = sentence + text + " "
            labels = labels + label + " "
        else:
            sentence = sentence[:-1]
            labels = labels[:-1]
            sentence_list.append((sentence, labels))
            sentence = ""
            labels = ""
    return sentence_list

class Vocabulary:
    def __init__(self, is_labels=False):
        self.is_labels = is_labels
        self.itos = {}
        if not is_labels:
            self.itos = {0: '<UNK>', 1: '<SOS>', 2: '<EOS>'}
        # initiate the token to index dict
        self.stoi = {k: j for j, k in self.itos.items()}

        self.freq_threshold = 0#1

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):  # split on space and converts the sentence to list of words
        return [tok.strip() for tok in text.split(' ')]

    def build_vocabulary(self, data, is_sentence=True):
        frequencies = {}  # init the freq dict
        idx = 0 if self.is_labels else 3  # index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk

        # calculate freq of words
        for sentence in data:
            for word in self.tokenizer(sentence):
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        # limit vocab by removing low freq words
        frequencies = {k: v for k, v in frequencies.items() if v > self.freq_threshold}

        # limit vocab to the max_size specified
        # frequencies = dict(
        #     sorted(frequencies.items(), key=lambda x: -x[1]))

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
            else:  # out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi['<UNK>'])

        return numericalized_text
    
    # TODO: check this method
    def to_text(self, numericalized_text):
        return self.itos[numericalized_text.item()] if self.is_labels else self.itos[numericalized_text[2].item()]

def sentence_to_windows(vocab, vocab_labels, sentence, labels=None, window_size=5):
    numerized_senetnce = [vocab.stoi["<SOS>"], vocab.stoi["<SOS>"]]
    numerized_senetnce += vocab.numericalize(sentence)
    numerized_senetnce.extend([vocab.stoi["<EOS>"]]*2)
    windows_sentence = [(numerized_senetnce[i - 2], numerized_senetnce[i - 1], numerized_senetnce[i],
                        numerized_senetnce[i + 1], numerized_senetnce[i + 2]) for i in
                        range(2, len(numerized_senetnce) - 2)]
    if labels:
        numerized_label = vocab_labels.numericalize(labels)
        return windows_sentence, numerized_label
    else:
        return windows_sentence
    

def data_to_window(vocab, vocab_labels, data, window_size=5):
    windows_sentences = []
    windows_labels = []
    sentences, labels = zip(*data)
    for sentence, tags in zip(sentences, labels):
        windows, numerized_tags = sentence_to_windows(vocab, vocab_labels, sentence, tags)
        windows_sentences.extend(windows)
        windows_labels.extend(numerized_tags)
    return windows_sentences, windows_labels

def plot_results(train_loss, val_loss, train_acc, val_acc):
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
    ax[1].plot(
        train_acc, label="train", color="red", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[1].plot(
        val_acc, label="val", color="blue", linestyle="--", linewidth=2, alpha=0.5
    )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()


train_data = read_data('ner/train', '\t')
sentences, labels = zip(*train_data)
vocab = Vocabulary()
vocab_labels = Vocabulary(is_labels=True)
vocab.build_vocabulary(sentences)
vocab_labels.build_vocabulary(labels)

dev_data = read_data('ner/dev', '\t')
dev_sentences, dev_labels = zip(*dev_data)