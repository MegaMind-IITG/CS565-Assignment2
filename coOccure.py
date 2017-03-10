import time
import pickle
from collections import Counter, defaultdict
import codecs
import itertools
from cytoolz import itertoolz
import os
import sys

from random import normalvariate
from math import sqrt
import spacy
import textacy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.random_projection import sparse_random_matrix
import networkx as nx 
# import svd
from numpy.linalg import norm


UNKNOWN = 'UNK'
WINDOW_WIDTH = 5
# data = codecs.open('data/text8', mode='r', encoding='utf-8')
# data = data.read().replace('\n', '')

def semantic_network_to_adj_matrix(graph):
    print 'Converting semantic network to adjacency matrix ...     '
    start = time.time()
    nodelist = [str(i) for i in range(10000)]
    coMatrix = nx.adjacency_matrix(graph, nodelist=nodelist)
    end = time.time()
    print("Took %d time to obtain co-occurence matrix from semantic network graph" % ((end - start)))
    return coMatrix

def terms_to_semantic_network(ordered_list_of_tokens, window_width = WINDOW_WIDTH):
    print 'Converting tokenised text8 to semantic network ... '
    if not isinstance(ordered_list_of_tokens[0], unicode): 
        ordered_list_of_tokens = map(unicode, ordered_list_of_tokens)
    
    start = time.time()
    graph = textacy.network.terms_to_semantic_network(ordered_list_of_tokens, window_width=window_width)
    end = time.time()
    print("Took %d time to convert text8 to semantic network graph" % ((end - start)))
    return graph

def terms_to_adj_matrix(data, window_width = WINDOW_WIDTH):
    # coMatrix[i][j] denotes number of times j has appeared in context of i
    print "Converting tokens to word-word cooccurence matrix ... "

    contexts = list(itertoolz.sliding_window(window_width, data))
    coMatrix = csr_matrix((10000, 10000), dtype=np.int64).toarray()
    for c in contexts:
        for pair in itertools.permutations(c, 2):
            coMatrix[pair[0]][pair[1]] = coMatrix[pair[0]][pair[1]] + 1
    coMatrix = coMatrix/2
    print("Done! ")
    return coMatrix


def load_dataset():
    print 'Reading preprocessed data ...'
    with open('./data/train_data.pickle', 'rb') as handle:
        dictionary = pickle.load(handle)
        reverse_dictionary = pickle.load(handle)
        data = pickle.load(handle)
        count = pickle.load(handle)
    return dictionary, reverse_dictionary, data, count

if __name__ == "__main__":
    # Loading the preprocessed data
    dictionary, reverse_dictionary, data, count = load_dataset()

    # Using Library to calculate the co-occurence matrix
    graph = terms_to_semantic_network(data)
    libCoMatrix = semantic_network_to_adj_matrix(graph)

    # Co-occurence matrix implemented from scratch
    codedCoMatrix = terms_to_adj_matrix(data)

    pickle.dump(codedCoMatrix, open('data/codedCoMatrix.p', 'w'))
