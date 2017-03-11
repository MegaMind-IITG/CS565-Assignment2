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


## Word2Vec

The CBOW model with negative sampling proposed by [Mikolov et al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) is implemented in Tensorflow. It can be run using `python part3.py`.

For negative sampling, the **count** of all the words in the corpus is passed to the model, which then generates a unigram distribution and uses the inbuilt NCE loss function as follows.

> neg_sample = tf.nn.fixed_unigram_candidate_sampler(y_in, 1, neg_sample_size, True, dictionary_size, unigrams=count)
> self.losses = tf.nn.nce_loss(W, b, tf.to_float(y_in), x_emb_in, neg_sample_size, dictionary_size, sampled_values=neg_sample)

Implementation details:
* Batch size = 5000
* Window size = 5
* Number of epochs = 50

The corpus was processed using `python preprocess.py` before training.
The obtained embeddings are saved in *embeddings/emb_w2v_50epochs.txt*

## Evaluations of word embeddings

### Named Entity Recognition (NER)