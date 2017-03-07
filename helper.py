import numpy as np

def readWordEmb(word_dict, fname, embSize=50):
	print "Reading word vectors"
	wv = []
	wl = []
	with open(fname, 'r') as f:
		for line in f :			
			vs = line.split()
			if len(vs) < 50 :
				continue
			vect = map(float, vs[1:])
			wv.append(vect)
			wl.append(vs[0])
	wordemb = []
	count = 0
	for word, id in word_dict.iteritems():
		if str(word) in wl:
			wordemb.append(wv[wl.index(str(word))])
		else:
			count += 1
			wordemb.append(wv[wl.index('UNK')])
	wordemb = np.asarray(wordemb, dtype='float32')
	print "Number of unknown words in word embedding", count
	return wordemb