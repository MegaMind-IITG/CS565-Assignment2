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
import svd
from numpy.linalg import norm
import Q2
import Q4

def lib_SVD(matrix, n_iter = 10):
    kwargs = {'n_iter' : n_iter}
    
    print "Starting the process SVD ..."
    start = time.time()
    model = textacy.tm.topic_model.TopicModel('lsa', n_topics = 50, **kwargs)
    model.fit(matrix)
    ans = model.transform(matrix)
    end  = time.time()
    print("Truncated vectors obtained. Took %d time. " % ((end - start)))
    return ans


def self_SVD(matrix, n_iter = 10):
    print "Using power method for SVD implementation ... "
    # m, n = matrix.shape
    # U = matrix
    # U_2 = np.square(matrix)
    # V = np.identity(n)
    # N_2 = U_2.sum()
    # s = 0
    # first = True
    # epsilon = 1e-10
    # while(math.sqrt(s) < (epsilon**2)*N_2 and not first):
    #     s = 0
    #     first = False
    #     for i in range(1, n):
    #         for j in range(i+1, n+1):
    #             s = s + (np.multiply(U[:, i], U[:, j]).sum())**2
    #             c = 

dictionary, reverse_dictionary, data, count = Q2.load_dataset()

with open('data/codedCoMatrix.p', 'r') as matrix:
    X = pickle.load(matrix)

ans = lib_SVD(X)
# ans2 = self_SVD(X)


# Removing the top 50 most occurence words

X = np.delete(X, [i for i in range(50)], axis = 0)
X = np.delete(X, [i for i in range(50)], axis = 1)

# Dictionary and reverse dictionary as per the edited vocab set
id2term = {i - 50:reverse_dictionary[i] for i in range(10000)}
term2id = {id2term[i]:i for i in id2term.keys()}

print "Starting Truncated SVD ... "
kwargs = {'n_iter' : 10}
model = textacy.tm.topic_model.TopicModel('lsa', n_topics = 50, **kwargs)
model.fit(X)
ans = model.transform(X)

# Contains projections of all the top 20 words for each topic
cluster_projections = np.ndarray(shape = (50,20,2), dtype = float)

# Dictionary from topic to list of 20 terms
topic2terms = {}
for topic_idx, top_terms in model.top_topic_terms(id2term, top_n = 20, topics=[i for i in range(50)]):
    # print('topic', topic_idx, ':', '   '.join(top_terms))
    topic2terms[topic_idx] = top_terms

# Dictionary from topic to list of ids of 20 terms 
topic2ids = {i:[term2id[j] for j in topic2terms[i]] for i in topic2terms.keys()}

print "Starting to calculate the projections ... "
tsne_model = TSNE(n_components = 2, random_state = 0)   
np.set_printoptions(suppress = True)
projections = tsne_model.fit_transform(ans)
print "Projections obtained."

# Extracting out projection from projection matrix of shape  (9950, 2) and copying them to cluster_projections matrix
for i in topic2ids.keys():
    top_terms_embedding = projections[topic2ids[i], :]
    cluster_projections[i] = top_terms_embedding

Q4.project_words(projections, term2id)

# Manually chosen out of 50 topics
best_topics = [1,2,6,10,12,29,33,39,41]

fig = plt.figure()
gs = gridspec.GridSpec(6, 3)
ax1 = fig.add_subplot(gs[0,0])
ax1.scatter(cluster_projections[1,0:5,0], cluster_projections[1,0:5,1], marker='x', color=colors[0])
ax2 = fig.add_subplot(gs[0,1])
ax2.scatter(cluster_projections[2,0:5,0], cluster_projections[2,0:5,1], marker='o', color=colors[0])
ax3 = fig.add_subplot(gs[0,2])
ax3.scatter(cluster_projections[6,0:5,0], cluster_projections[6,0:5,1], marker='x', color=colors[1])
ax4 = fig.add_subplot(gs[1,0])
ax4.scatter(cluster_projections[10,0:5,0], cluster_projections[10,0:5,1], marker='o', color=colors[1])
ax5 = fig.add_subplot(gs[1,1])
ax5.scatter(cluster_projections[12,0:5,0], cluster_projections[12,0:5,1], marker='x', color=colors[2])
ax6 = fig.add_subplot(gs[1,2])
ax6.scatter(cluster_projections[29,0:5,0], cluster_projections[29,0:5,1], marker='o', color=colors[2])
ax7 = fig.add_subplot(gs[2,0])
ax7.scatter(cluster_projections[33,0:5,0], cluster_projections[33,0:5,1], marker='x', color=colors[3])
ax8 = fig.add_subplot(gs[2,1])
ax8.scatter(cluster_projections[39,0:5,0], cluster_projections[39,0:5,1], marker='o', color=colors[3])
ax9 = fig.add_subplot(gs[2,2])
ax9.scatter(cluster_projections[41,0:5,0], cluster_projections[41,0:5,1], marker='x', color=colors[4])

ax10 = fig.add_subplot(gs[3:,:])
ax10.scatter(cluster_projections[1,0:5,0], cluster_projections[1,0:5,1], marker='x', color=colors[0])
ax10.scatter(cluster_projections[2,0:5,0], cluster_projections[2,0:5,1], marker='o', color=colors[0])
ax10.scatter(cluster_projections[6,0:5,0], cluster_projections[6,0:5,1], marker='x', color=colors[1])
ax10.scatter(cluster_projections[10,0:5,0], cluster_projections[10,0:5,1], marker='o', color=colors[1])
ax10.scatter(cluster_projections[12,0:5,0], cluster_projections[12,0:5,1], marker='x', color=colors[2])
ax10.scatter(cluster_projections[29,0:5,0], cluster_projections[29,0:5,1], marker='o', color=colors[2])
ax10.scatter(cluster_projections[33,0:5,0], cluster_projections[33,0:5,1], marker='x', color=colors[3])
ax10.scatter(cluster_projections[39,0:5,0], cluster_projections[39,0:5,1], marker='o', color=colors[3])
ax10.scatter(cluster_projections[41,0:5,0], cluster_projections[41,0:5,1], marker='x', color=colors[4])

text = iter(['1', '2', '6', '10', '12', '29', '33', '39', '41', '1', '2', '6', '10', '12', '29', '33', '39', '41'])

gs.update(wspace=0.5, hspace=0.5)

fig = plt.gcf()

plotly_fig = tls.mpl_to_plotly( fig )
for dat in plotly_fig['data']:
    t = text.next()
    dat.update({'name': t, 'text':t})

plotly_fig['layout']['title'] = 'Subplots with variable widths and heights'
plotly_fig['layout']['margin'].update({'t':40})
plotly_fig['layout']['showlegend'] = True
plot_url = py.plot(plotly_fig, filename='mpl-subplot-variable-width')