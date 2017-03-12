# CS565-Assignment2

## Neural Probabilistic Language Model

The model proposed by [Bengio et al.](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) is implemented in Tensorflow. It can be run using `python part1.py`.

Implementation details:
* Batch size = 5000
* Window size = 5
* Number of epochs = 5

The corpus was processed using `python preprocess.py` before training.
The obtained embeddings are saved in *embeddings/emb_nplm_5epochs.txt*

## Singular Value Decomposition

There are four main file associated with Part 2:
* `coOccure.py` : This file will load the pre-processed data obtained from `preprocess.py` and obtain a co-occurence matrix from the ordered list of tokens. There is a global variable named `WINDOW_WIDTH` that can be changed to adjust the size of the context window. It contains both types of implementation: using a library and implemenation from scratch. 
* `svd.py` : This file is run after obtaining the co-occurrence matrix. It taken an argument to indicated whether you want to run the obtain the truncated SVD using library method or the method implemented fom scratch. 
* `self_svd.py` : This contains the code for truncated SVD that was implemented from scratch. It implemented the algorithm proposed in [Bentbib_A.H_Kander_A](http://www.anstuocmath.ro/mathematics/anale2015vol2/Bentbib_A.H.__Kanber_A..pdf)
* `tSNE.py` : This file takes projections as input and project vectors on 2-D plane.  

## Word2Vec

The CBOW model with negative sampling proposed by [Mikolov et al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) is implemented in Tensorflow. It can be run using `python part3.py`.

For negative sampling, the **count** of all the words in the corpus is passed to the model, which then generates a unigram distribution and uses the inbuilt NCE loss function as follows.

```python
neg_sample = tf.nn.fixed_unigram_candidate_sampler(y_in, 1, neg_sample_size, True, dictionary_size, unigrams=count)
self.losses = tf.nn.nce_loss(W, b, tf.to_float(y_in), x_emb_in, neg_sample_size, dictionary_size, sampled_values=neg_sample)
```

Implementation details:
* Batch size = 5000
* Window size = 5
* Number of epochs = 50

The corpus was processed using `python preprocess.py` before training.
The obtained embeddings are saved in *embeddings/emb_w2v_50epochs.txt*

## Evaluations of word embeddings

### Using t-SNE projections

We chose the words between index 100 and 200 to draw t_SNE projections. It can be run using `python part4_1.py`.

Three plots are generated - *part1.png* for NPLM Model, *part2.png* for LSA model and *part3.png* for CBOW model. We also generate the closest embedding for each of the 100 words. *scikit learn* package is used for t-SNE projection.

Implementation details:
* No. of components: 2 (Dimension of the embedded space)
* Perplexity: 30
* No. of iterations: 5000

### Named Entity Recognition (NER)

#### Generating the dataset

Required files:
* enwik8
* convert.pl
* extract.py

We use the [enwik8](http://mattmahoney.net/dc/textdata.html) corpus consisting of the first 10^8 words in the English language Wikipedia. This file contains complex XML encoding, so it is first converted to English sentences using a Perl script as `perl data/convert.pl data/enwik8 > data/text8`.

After obtaining the corpus, it is converted to the CoNLL format using the POS tagging, Chunk tagging and NER tagging modules provided in NLTK as `python data/extract.py`. 

#### NER model

The data in CoNLL format is first processed into lists and Numpy arrays using `python part4-ner/preprocess.py`. The neural network model is implemented in *model.py* and the training is done in *main.py*.

*model.py* contains code for a single hidden layer model and a CNN model. In addition, an LSTM model was also implemented, but did not perform well.
