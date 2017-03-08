from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from helper import saveWordEmb

import collections
import math
import zipfile

import numpy as np
import pickle
import tensorflow as tf

class Word2vec(object):

	def __init__(self, dictionary_size, count, emb_size=50, window_size=5, neg_sample_size=10, learning_rate=0.01, l2_reg_lambda=0.01):

		self.input  = tf.placeholder(tf.int32, [None,window_size], name="input")
		self.output = tf.placeholder(tf.int32, [None], name="output")
		
		# Initialization
		self.W_emb =  tf.Variable(tf.random_uniform([dictionary_size, emb_size], -1.0, +1.0))
		
		# Embedding layer
		x_emb = tf.nn.embedding_lookup(self.W_emb, self.input)	
		x_emb_in = tf.reduce_mean(x_emb,axis=1,keep_dims=False)

		# Hidden layer
		W = tf.Variable(tf.truncated_normal([dictionary_size, emb_size], stddev=0.1), name="W")
		b = tf.Variable(tf.constant(0.1, shape=[dictionary_size]), name="b")

		# Negative sampling
		y_in = tf.expand_dims(tf.to_int64(self.output),axis=-1)
		neg_sample = tf.nn.fixed_unigram_candidate_sampler(y_in, 1, neg_sample_size, True, 
			dictionary_size, unigrams=count)

		l2_loss = tf.constant(0.0)
		l2_loss += tf.nn.l2_loss(W)

		# prediction and loss function
		self.losses = tf.nn.nce_loss(W, b, tf.to_float(y_in), x_emb_in, neg_sample_size, dictionary_size, sampled_values=neg_sample)
		self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda*l2_loss

		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)  

		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

		self.sess.run(tf.global_variables_initializer())

	def train_step(self, in_batch, out_batch):
    		feed_dict = {
				self.input: in_batch,
				self.output: out_batch
	    			}
   		_,loss, W_e = self.sess.run([self.optimizer, self.loss, self.W_emb], feed_dict)
    		# print ("step "+str(step) + " loss "+str(loss))
    		return W_e,loss


with open('./data/train_data.pickle', 'rb') as handle:
	dictionary = pickle.load(handle)
	reverse_dictionary = pickle.load(handle)
	data = pickle.load(handle)
	count = pickle.load(handle)

unigram_counts = []
for word in count:
	unigram_counts.append(word[1])

w2v = Word2vec(len(dictionary),unigram_counts)
num_epochs = 50
num_train = len(data) - 4
batch_size = 5000
num_batches = num_train//batch_size
window = 5

W = []

for j in range(num_epochs):
	for i in range(num_batches):
		x_batch = []
		y_batch = []
		for k in range(batch_size):
			temp_list = []
			for p in range(window):
				temp_list.append(data[i*batch_size+k+p])
		
			x_batch.append(temp_list)	
			y_batch.append(data[i*batch_size+k+window])

		W, loss = w2v.train_step(x_batch,y_batch)
		print("Epoch",j+1,"Batch",i+1,"Loss",loss)


if(saveWordEmb("emb_w2v_50epochs.txt",count,W)):
	print("Embeddings saved to file.\n")