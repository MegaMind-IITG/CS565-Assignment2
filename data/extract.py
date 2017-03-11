import nltk
import re

TRAIN = 950000
TEST = 40000

fin = open("./text8-caps.sample", "r")
f_train = open("./text8.train","w")
f_test = open("./text8.test","w")

data = fin.read()
data = ''.join([i if ord(i) < 128 else ' ' for i in data])

print "data read"

sentences = nltk.sent_tokenize(data)
print "sentences tokenized"
sentences = [nltk.word_tokenize(sent) for sent in sentences]
print "words tokenized"
sentences = [nltk.pos_tag(sent) for sent in sentences]
print "POS tagging done"

grammar = r"""
  NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
      {<NNP>+}                # chunk sequences of proper nouns
"""
cp = nltk.RegexpParser(grammar)
sentences1 = [nltk.tree2conlltags(cp.parse(sent)) for sent in sentences]
print "Chunk tagging done"
sentences2 = [nltk.tree2conlltags(nltk.ne_chunk(sent)) for sent in sentences]
print "NER tagging done"

final = []
for cs,ns in zip(sentences1,sentences2):
	for c,n in zip(cs,ns):
		x = list(c)
		x.append(n[2])
		final.append(x)


for i in final[0:TRAIN]:
	f_train.write("%s\n" % i)

for i in final[TRAIN+1:TRAIN+TEST]:
	f_test.write("%s\n" % i)


fin.close()
f_train.close()
f_test.close()