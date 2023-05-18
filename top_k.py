import numpy as np
from numpy.linalg import norm


def read_data(fname):
    words_list = {}
    with open(fname, encoding="utf8") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        word = line.strip()
        words_list[word] = vecs[idx]
    return words_list


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity


def most_similiar(word, k):
    query_vector = words[word]
    similarities = np.zeros(vecs.shape[0])
    for i, vector in enumerate(vecs):
        similarities[i] = cosine_similarity(query_vector, vector)
    indices = np.argsort(similarities)[::-1][1:k + 1]
    top_k_similar_vectors = vecs[indices]
    return top_k_similar_vectors, indices, similarities[indices]


vecs = np.loadtxt("wordVectors.txt")
words = read_data("vocab.txt")
words_to_check = ['dog', 'england', 'john', 'explode', 'office']

for word in words_to_check:
    sim, ind, dist = most_similiar(word, 5)
    print(word)
    [print(list(words.keys())[word], dist[i]) for i, word in enumerate(ind)]
    print('___________________________')