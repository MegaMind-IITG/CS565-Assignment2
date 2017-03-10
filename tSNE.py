import numpy as numpy
import spacy
import nltk
from nltk.corpus import inaugural
import textacy
import time
import pickle
import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.plotly as py
import plotly.tools as tls
import codecs
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
import helper

vocab = []
colors = ['b', 'c', 'y', 'm', 'r']


def project_words(projections, term2id = None):
    if not term2id:
        print "Error: Give the dictionary as an argument, i.e. mapping from term to ID"
        return
    ifp = open('words.tsne', 'r')
    words = ifp.readlines()
    words = [unicode(i.strip(), 'utf-8') for i in words]
    vectors = np.array([projections[term2id[i]] for i in words])
    X = vectors[:, 0]
    Y = vectors[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    for i, txt in enumerate(words):
        ax.annotate(txt,(X[i], Y[i]))
    plt.show()
    ifp.close()

def project_topics(topic_list, cluster_projections, annotation_list = None):
    fig = plt.figure()
    dim = int(sqrt(len(topic_list))) + 1
    gs = gridspec.GridSpec(dim, dim)
    counter = 0
    for i in range(dim):
        for j in range(dim):
            ax = fig.add_subplot(gs[i,j])
            ax.scatter(cluster_projections[topic_list[counter],:,0], cluster_projections[topic_list[counter],0:5,1])

    gs.update(wspace=0.5, hspace=0.5)
    fig = plt.gcf()
    plotly_fig = tls.mpl_to_plotly( fig )
    for dat in plotly_fig['data']:
        t = annotation_list.next()
        dat.update({'name': t, 'text':t})

    plotly_fig['layout']['title'] = 'Plot'
    plotly_fig['layout']['margin'].update({'t':40})
    plotly_fig['layout']['showlegend'] = True
    plot_url = py.plot(plotly_fig, filename='Plot')

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print "Error: \"python tSNE.py <file containing embeddings> <file containing dictionary>\". Exiting ... "
        sys.exit()
    
    with open(sys.argv[2], 'r') as file:
        dictionary = pickle.load(file)
        
    with open(sys.argv[1], 'r') as matrix:
        X = helper.readWordEmb(dictionary, matrix)

    print "Loaded all data."

    print "Starting to calculate the projections ... "
    tsne_model = TSNE(n_components = 2, random_state = 0)   
    np.set_printoptions(suppress = True)
    projections = tsne_model.fit_transform(X)
    print "Projections obtained."


    #### Edit this part
    project_words(projections, dictionary)