from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import collections
import math
import zipfile

import numpy as np
import pickle
import tensorflow as tf

from model import *

TRAIN_FILE = '../data/text8_train.pickle'
TEST_FILE = '../data/text8_test.pickle'
EMB_FILE = '../data/w2v.pickle'


with open(TRAIN_FILE, 'rb') as handle:
	train_words = pickle.load(handle)
	train_pos = pickle.load(handle)
	train_chunk = pickle.load(handle)
	train_capital = pickle.load(handle)
	train_tags = pickle.load(handle)

with open(TEST_FILE, 'rb') as handle:
	test_words = pickle.load(handle)
	test_pos = pickle.load(handle)
	test_chunk = pickle.load(handle)
	test_capital = pickle.load(handle)
	test_tags = pickle.load(handle)


with open(EMB_FILE, 'rb') as handle:
	we = pickle.load(handle)


train_tags_num = [0,0,0,0,0]
test_tags_num = [0,0,0,0,0]

for i in train_tags:
	train_tags_num[i] += 1

for i in test_tags:
	test_tags_num[i] += 1

"""

tag_dict_size = 5
pos_dict_size = 5
chunk_dict_size = 5
capital_dict_size = 2
word_dict_size = len(we)
ner_dict_size = 5 


context = 5
num_epochs = 20
N = 1
num_train = len(train_words) - 5
batch_size = 2000
num_batches = num_train//batch_size
window = 11

fp = open('./results/ner_w2v_with_update.txt','wb')
conf = open('./results/confusion_ner_w2v_with_update.txt','wb')

ner = NER(we, word_dict_size, pos_dict_size, chunk_dict_size, capital_dict_size)

def test_step(W_te, pos_te, chunk_te, capital_te, y_te):
	w_batch = []
	pos_batch = []
	chunk_batch = []
	capital_batch = []
	y_batch = []
	for i in range(len(W_te)-window):
		w = []
		pos = []
		chunk = []
		capital = []
		for p in range(window):
			w.append(W_te[i+p])
			pos.append(pos_te[i+p])
			chunk.append(chunk_te[i+p])
			capital.append(capital_te[i+p])
		w_batch.append(w)
		pos_batch.append(pos)
		chunk_batch.append(chunk)
		capital_batch.append(capital)	
		y_batch.append(y_te[i+context])

	a,p = ner.test_step(w_batch,pos_batch,chunk_batch,capital_batch,y_batch)
	return p


for j in range(num_epochs):
	for i in range(num_batches):
		w_batch = []
		pos_batch = []
		chunk_batch = []
		capital_batch = []
		y_batch = []
		for k in range(batch_size):
			w = []
			pos = []
			chunk = []
			capital = []
			for p in range(window):
				w.append(train_words[i*batch_size+k+p])
				pos.append(train_pos[i*batch_size+k+p])
				chunk.append(train_chunk[i*batch_size+k+p])
				capital.append(train_capital[i*batch_size+k+p])
			w_batch.append(w)
			pos_batch.append(pos)
			chunk_batch.append(chunk)
			capital_batch.append(capital)	
			y_batch.append(train_tags[i*batch_size+k+context])

		loss, accuracy = ner.train_step(w_batch,pos_batch,chunk_batch,capital_batch,y_batch)
		print("Epoch",j+1,"Batch",i+1,"Loss",loss,"Accuracy",accuracy)

	if((j+1)%N==0):
		pred = test_step(test_words,test_pos,test_chunk,test_capital,test_tags)		 
		print ("Test data size ", len(pred))
		y_true = test_tags[context:len(test_tags)-(window - context)]
		y_pred = pred
		print(confusion_matrix(y_true,y_pred))
		print(str(precision_score(y_true, y_pred, average='weighted' )))
		print(str(recall_score(y_true, y_pred, average='weighted' )))
		print(str(f1_score(y_true, y_pred, average='weighted' )))
		fp.write(str(precision_score(y_true, y_pred, average='weighted' )))
		fp.write('\t')
		fp.write(str(recall_score(y_true, y_pred, average='weighted' )))
		fp.write('\t')
		fp.write(str(f1_score(y_true, y_pred, average='weighted' )))
		fp.write('\t')
		fp.write('\n')
		conf.write(str(j+1))
		conf.write('\n')
		conf.write(str(confusion_matrix(y_true,y_pred)))
		conf.write('\n\n')

fp.close()
"""