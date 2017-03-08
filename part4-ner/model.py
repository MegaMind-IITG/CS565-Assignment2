import numpy as np
import tensorflow as tf


class NER(object):

	def __init__(self, we, word_dict_size, pos_dict_size, chunk_dict_size, capital_dict_size, tag_dict_size=5, word_emb_size=50, pos_emb_size=5, chunk_emb_size=5, capital_emb_size=5, window_size=5, hidden_layer_size=100, learning_rate=0.01, l2_reg_lambda=0.01):

		self.w  = tf.placeholder(tf.int32, [None,window_size])
		self.pos = tf.placeholder(tf.int32, [None,window_size])
		self.chunk = tf.placeholder(tf.int32, [None,window_size])
		self.capital = tf.placeholder(tf.int32, [None,window_size])
		self.tags = tf.placeholder(tf.int32, [None])
		
		# Initialization
		self.W_emb =  tf.Variable(we, trainable=False)
		self.pos_emb = tf.Variable(tf.random_uniform([pos_dict_size, pos_emb_size], -1.0, +1.0))
		self.chunk_emb = tf.Variable(tf.random_uniform([chunk_dict_size, chunk_emb_size], -1.0, +1.0))
		self.capital_emb = tf.Variable(tf.random_uniform([capital_dict_size, capital_emb_size], -1.0, +1.0))
		
		# Embedding layer
		emb0 = tf.nn.embedding_lookup(self.W_emb, self.w)
		emb1 = tf.nn.embedding_lookup(self.pos_emb, self.pos)
		emb2 = tf.nn.embedding_lookup(self.chunk_emb, self.chunk)
		emb3 = tf.nn.embedding_lookup(self.capital_emb, self.capital)

		# x_emb = tf.concat([emb0, emb1, emb2, emb3],2)
		x_emb = emb0
		# emb_size = word_emb_size + pos_emb_size + chunk_emb_size + capital_emb_size
		emb_size = word_emb_size

		x_emb = tf.reshape(x_emb,[-1,window_size*emb_size])

		
		y_one_hot = tf.one_hot(self.tags,tag_dict_size)
		

		# print (x_emb.get_shape())
		# print (y_emb.get_shape())

		# Hidden layer
		H = tf.Variable(tf.truncated_normal([window_size*emb_size, hidden_layer_size], stddev=0.1), name="H")
		d = tf.Variable(tf.constant(0.1, shape=[hidden_layer_size]), name="d")
		h1 = tf.tanh(tf.nn.xw_plus_b(x_emb, H, d))

		#Output layer
		U = tf.Variable(tf.truncated_normal([hidden_layer_size, tag_dict_size], stddev=0.1), name="U")
		b = tf.Variable(tf.constant(0.1, shape=[tag_dict_size]), name="b")
		
		h2 = tf.nn.relu(tf.nn.xw_plus_b(h1, U, b))


		l2_loss = tf.constant(0.0)
		l2_loss += tf.nn.l2_loss(H)
		l2_loss += tf.nn.l2_loss(U)


		# prediction and loss function
		self.predictions = tf.argmax(h2, 1)
		self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=h2, labels=y_one_hot)
		self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda*l2_loss

		# Accuracy
		self.correct_predictions = tf.equal(self.predictions, tf.argmax(y_one_hot, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"))	

		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		self.sess = tf.Session(config=session_conf)  

		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

		self.sess.run(tf.global_variables_initializer())

	def train_step(self, W_batch, pos_batch, chunk_batch, capital_batch, y_batch):
			feed_dict = {
				self.w: W_batch, 
				self.pos: pos_batch,
				self.chunk: chunk_batch,
				self.capital: capital_batch,
				self.tags: y_batch 
					}
			_, loss, accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict)
			return loss, accuracy

	def test_step(self, W_batch, pos_batch, chunk_batch, capital_batch, y_batch):
			feed_dict = {
				self.w: W_batch, 
				self.pos: pos_batch,
				self.chunk: chunk_batch,
				self.capital: capital_batch,
				self.tags: y_batch 
					}
			loss, accuracy, predictions = self.sess.run([self.loss, self.accuracy, self.predictions], feed_dict)
			#print "Accuracy in test data", accuracy
			return accuracy, predictions