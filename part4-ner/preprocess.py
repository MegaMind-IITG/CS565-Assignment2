from __future__ import print_function
import pickle as pickle
import sys


def pos(tag):
	if tag == 'NN' or tag == 'NNS':
		return 0
	elif tag == 'FW':
		return 1
	elif tag == 'NNP' or tag == 'NNPS':
		return 2
	elif 'VB' in tag:
		return 3
	else:
		return 4


def chunk(tag):
	if 'NP' in tag:
		return 0
	elif 'VP' in tag:
		return 1
	elif 'PP' in tag:
		return 2
	elif tag == 'O':
		return 3
	else:
		return 4


def capital(word):
	if ord('A') <= ord(word[0]) <= ord('Z'):
		return 1
	else:
		return 0


def readWordEmb(fname, out_file):
	we = []
	with open(fname, 'r') as f:
		for line in f :			
			vs = line.split()
			if len(vs) < 50 :
				continue
			vect = map(float, vs[1:])
			we.append(vect)
	with open(out_file,'wb') as handle:
		pickle.dump(we,handle)


def getWordIndex(fname):
	w_ind = {}
	ind = 0
	with open(fname, 'r') as f:
		for line in f :			
			vs = line.split()
			if vs[0] not in w_ind:
				w_ind[vs[0]] = ind
				ind += 1
	return w_ind


def get_input(w_ind, input_file, out_file):
	print('processing %s' % input_file)
	text = []
	entity_tag = []
	pos_tag = []
	chunk_tag = []
	capital_tag = []
	
	with open(input_file,'r') as f:
		for line in f:
			if line in ['\n', '\r\n']:
				continue
			else:
				curword = line.split()[0]
				if curword in w_ind:
					word = w_ind[curword]
				else:
					word = w_ind['UNK']
				text.append(word)
				pos_tag.append(pos(line.split()[1]))  # adding pos embeddings
				chunk_tag.append(chunk(line.split()[2]))  # adding chunk embeddings
				capital_tag.append(capital(line.split()[0]))  # adding capital embedding
				
				t = line.split()[3]
				# Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc
				if t.endswith('PER'):
					entity_tag.append(0)
				elif t.endswith('LOC'):
					entity_tag.append(1)
				elif t.endswith('ORG'):
					entity_tag.append(2)
				elif t.endswith('MISC'):
					entity_tag.append(3)
				elif t.endswith('O'):
					entity_tag.append(4)
				else:
					print("error in input tag {%s}" % t)
					sys.exit(0)
	with open(out_file, 'wb') as handle:
		pickle.dump(text,handle)
		pickle.dump(pos_tag,handle)
		pickle.dump(chunk_tag,handle)
		pickle.dump(capital_tag,handle)
		pickle.dump(entity_tag,handle)


TRAIN_FILE = "../data/eng.train"
DEV_FILE = "../data/eng.testa"
TEST_FILE = "../data/eng.testb"

word_emb_path = '../embeddings/emb_nplm_5epochs.txt'
w_ind = getWordIndex(word_emb_path)
get_input(w_ind, TRAIN_FILE, '../data/train_set.pickle')
get_input(w_ind, DEV_FILE, '../data/dev_set.pickle')
get_input(w_ind, TEST_FILE, '../data/test_set.pickle')

readWordEmb(word_emb_path,'../data/nplm.pickle')
word_emb_path = '../embeddings/emb_w2v_50epochs.txt'
readWordEmb(word_emb_path,'../data/w2v.pickle')
