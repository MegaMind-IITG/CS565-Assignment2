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
	vectors = [projections[term2id[i]] for i in words]
	X = vectors[0, :]
	Y = vectors[1, :]
	fig, ax = plt.subplots()
	ax.scatter(X, Y)
	for i, txt in enumerate(words):
		ax.annotate(txt,(X[i], Y[i]))
	plt.show()
	ifp.close()

with open('./data/train_data.pickle', 'rb') as handle:
	dictionary = pickle.load(handle)
	reverse_dictionary = pickle.load(handle)
	data = pickle.load(handle)
	count = pickle.load(handle)

if len(sys.argv) < 2:
	print "Error: Give the file for projection matrix as an argument. Exiting ... "
	sys.exit()

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