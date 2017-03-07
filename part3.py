from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import zipfile

import numpy as np
import pickle
import tensorflow as tf

class Word2vec(object):

	def __init__(self, dictionary_size, count, emb_size=50, window_size=5, neg_sample_size=10, learning_rate=0.01, l2_reg_lambda=0.01):

		self.input  = tf.placeholder(tf.int32, [window_size], name="input")
		self.output = tf.placeholder(tf.float32, name="output")
		
		# Initialization
		self.W_emb =  tf.Variable(tf.random_uniform([dictionary_size, emb_size], -1.0, +1.0))
		
		# Embedding layer
		x_emb = tf.nn.embedding_lookup(self.W_emb, self.input)	
		x_emb_in = tf.reduce_mean(x_emb,axis=0,keep_dims=True)

		# print (x_emb.get_shape())
		# print (y_emb.get_shape())

		# Hidden layer
		W = tf.Variable(tf.truncated_normal([dictionary_size, emb_size],stddev=1.0 / math.sqrt(emb_size)))
		b = tf.Variable(tf.zeros([dictionary_size]))

		# Negative sampling
		neg_sample = tf.nn.fixed_unigram_candidate_sampler(tf.expand_dims(tf.to_int64(self.output),axis=0), 1, neg_sample_size, True, 
			dictionary_size, distortion=1.0, num_reserved_ids=0, num_shards=1, shard=0, unigrams=count, seed=None, name=None)

		l2_loss = tf.constant(0.0)
		l2_loss += tf.nn.l2_loss(W)

		# prediction and loss function
		self.losses = tf.nn.nce_loss(W, b, x_emb_in, tf.expand_dims(tf.convert_to_tensor(self.output),axis=-1), neg_sample_size, dictionary_size, sampled_values=neg_sample)
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
no_of_epochs=5

W = []
total_loss = 0
for j in xrange(no_of_epochs):
	for i in xrange(len(data)-4):
		temp_list=[data[i], data[i+1], data[i+2], data[i+3], data[i+4]]
		tmp_in=np.array(temp_list,dtype=np.int32)
		tmp_out=data[i+5]
		W, loss = w2v.train_step(tmp_in,tmp_out)
		total_loss += loss
		if(i%1000==0):
			print("Epoch",j+1,"Token",i,"Average loss",total_loss/1000)
			total_loss = 0