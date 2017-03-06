from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import zipfile

import numpy as np
import pickle
import tensorflow as tf

class NPLM(object):

	def __init__(self, dictionary_size, emb_size=50, window_size=5, hidden_layer_size=100, learning_rate=0.01, l2_reg_lambda=0.01):

		self.input  = tf.placeholder(tf.int32, [window_size], name="input")
		self.output = tf.placeholder(tf.int32, [1], name="output")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		
		# Initialization
		W_emb =  tf.Variable(tf.random_uniform([dictionary_size, emb_size], -1.0, +1.0))
		
		# Embedding layer
		x_emb = tf.nn.embedding_lookup(W_emb, self.input)				
		y_emb = tf.nn.embedding_lookup(W_emb, self.output)

		x_emb = tf.reshape(x_emb,[-1])
		x_emb_expanded = tf.expand_dims(x_emb,axis=0)

		# print (x_emb.get_shape())
		# print (y_emb.get_shape())

		# Fully connetected layer
		H = tf.Variable(tf.truncated_normal([window_size*emb_size, hidden_layer_size], stddev=0.1), name="H")
		d = tf.Variable(tf.constant(0.1, shape=[hidden_layer_size]), name="d")
		hidden_layer_output = tf.tanh(tf.nn.xw_plus_b(x_emb_expanded, H, d))

		#Regression layer
		U = tf.Variable(tf.truncated_normal([hidden_layer_size, dictionary_size], stddev=0.1), name="U")
		b = tf.Variable(tf.constant(0.1, shape=[dictionary_size]), name="b")
		W = tf.Variable(tf.truncated_normal([window_size*emb_size,dictionary_size],stddev=0.1), name="W")

		reg_output = tf.add(tf.nn.xw_plus_b(hidden_layer_output, U, b),tf.matmul(x_emb_expanded,W))

		#Softmax layer
		output_pred = tf.nn.softmax(reg_output)

		l2_loss = tf.constant(0.0)
		l2_loss += tf.nn.l2_loss(H)
		l2_loss += tf.nn.l2_loss(U)
		l2_loss += tf.nn.l2_loss(W)


		# prediction and loss function
		self.prediction = tf.argmax(output_pred, axis=[1], name="predictions")
		y_emb_pred = tf.nn.embedding_lookup(W_emb, self.prediction)

		self.loss = tf.norm(tf.sub(y_emb,y_emb_pred)) + l2_loss

		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)  

		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)

		self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

		self.sess.run(tf.initialize_all_variables())

	def train_step(self, in_batch, out_batch):
    		feed_dict = {
				self.input: in_batch,
				self.output: out_batch,
				self.dropout_keep_prob: 0.5,
	    			}
   		step, loss, W_emb = self.sess.run([self.global_step, self.loss, W_emb], feed_dict)
    		print ("step "+str(step) + " loss "+str(loss))
    		return W_emb




with open('./data/train_data.pickle', 'rb') as handle:
	dictionary = pickle.load(handle)
	reverse_dictionary = pickle.load(handle)
	data = pickle.load(handle)
	count = pickle.load(handle)


nplm = NPLM(len(dictionary))
no_of_epochs=1

W = []
for j in xrange(no_of_epochs):
	for i in xrange(len(data)-4):
		temp_list=[data[i], data[i+1], data[i+2], data[i+3], data[i+4]]
		tmp_in=np.array(temp_list,dtype=np.int32)
		tmp_out=data[i+5]
		W = nplm.train_step(tmp_in,tmp_out)

print (W[0])

