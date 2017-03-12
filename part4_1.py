from sklearn.manifold import TSNE
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import pickle
from scipy import spatial

#part1
filename="./embeddings/emb_nplm_5epochs.txt"
W = np.loadtxt(filename, delimiter=' ',  usecols=range(1,51))
f=open(filename,"r")
lines=f.readlines()
word_labels=[]
for x in lines:
    word_labels.append(x.split(' ')[0])
f.close()

no_of_words=100
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
new_emb = tsne.fit_transform(W[100:100+no_of_words, :])
labels = [word_labels[i] for i in xrange(100,100+no_of_words)]
plt.figure(figsize=(20, 20)) 
for index, label in enumerate(labels):
	x, y = new_emb[index, :]
	plt.scatter(x, y)
	plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
plt.savefig('part1.png')



tree = spatial.KDTree(W)
for i in xrange(100,200):
	_,l=tree.query(W[i],4)
	sim_words=[]
	for j in xrange(1,len(l)):
		sim_words.append(word_labels[l[j]])
	print word_labels[i],sim_words


###############################################################################
#part2
###############################################################################
with open('./embeddings/libSVD.embed', 'rb') as handle:
	arr = pickle.load(handle)

print "Part2 SVD"
a = arr[0][1]
for i in xrange(len(arr)):
	a=np.vstack((a, arr[i][1]))
W = np.delete(a, (0), axis=0)

no_of_words=100
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
new_emb = tsne.fit_transform(W[100:100+no_of_words, :])
labels = [word_labels[i] for i in xrange(100,100+no_of_words)]
plt.figure(figsize=(20, 20)) 
for index, label in enumerate(labels):
	x, y = new_emb[index, :]
	plt.scatter(x, y)
	plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
plt.savefig('part2.png')



tree = spatial.KDTree(W)
for i in xrange(100,200):
	_,l=tree.query(W[i],4)
	sim_words=[]
	for j in xrange(1,len(l)):
		sim_words.append(word_labels[l[j]])
	print word_labels[i],sim_words

###############################################################################
#part3
###############################################################################
print "Part3"
filename="./embeddings/emb_w2v_50epochs.txt"
W = np.loadtxt(filename, delimiter=' ',  usecols=range(1,51))
f=open(filename,"r")
lines=f.readlines()
word_labels=[]
for x in lines:
    word_labels.append(x.split(' ')[0])
f.close()

no_of_words=100
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
new_emb = tsne.fit_transform(W[100:100+no_of_words, :])
labels = [word_labels[i] for i in xrange(100,100+no_of_words)]
plt.figure(figsize=(20, 20)) 
for index, label in enumerate(labels):
	x, y = new_emb[index, :]
	plt.scatter(x, y)
	plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
plt.savefig('part3.png')



tree = spatial.KDTree(W)
for i in xrange(100,200):
	_,l=tree.query(W[i],4)
	sim_words=[]
	for j in xrange(1,len(l)):
		sim_words.append(word_labels[l[j]])
	print word_labels[i],sim_words




