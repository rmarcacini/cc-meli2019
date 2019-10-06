# Consensus Categorization (C²) for MercadoLivre Data Challenge 2019

The C² method is a supervised and transductive learning method based on the Consensus Clustering method that I investigated during my [doctorate at ICMC-USP](http://www.teses.usp.br/teses/disponiveis/55/55134/tde-05082015-094733/en.php).

Here, the C² method has been adapted to handle the dataset provided by [Meli Data Challenge 2019](https://ml-challenge.mercadolibre.com/).

In short, the C² method has the following steps:

* Preprocess product titles by removing stopwords (English, Portuguese, and Spanish), numbers, and special characters. Source: [meli/preprocess.py](meli/preprocess.py).
* Learn a textual representation for product titles by using [fasttext word embeddings](https://fasttext.cc/). This word embedding is useful for initializing classification models.
* Get different dataset samples, both by sampling instances and features. Source: [meli/sampling.py](meli/sampling.py)
* Get different classification models for each sampling. It is important that there is diversity in classification model solutions. Source: [meli/models.py](meli/models.py)
* Build a [heterogeneous network](https://ieeexplore.ieee.org/abstract/document/7536145/) with the following node types: product, terms, and classification models. Some network nodes are labeled considering the training set and the categories predicted by the classification models. The heterogeneous network is regularized through a [consensus function](https://dl.acm.org/citation.cfm?id=2983730) that will return to final categorization. Source: [meli/consensus.py](meli/consensus.py)

The C² method ranked fourth (private leaderboard) in the Meli Data Challenge 2019. It can be improved by either adding more classification models or tuning the consensus function.

# Requirements and Dependencies

* python 3
* numpy
* pandas
* keras
* gensim
* pickle
* tqdm
* sklearn
* networkx
* nltk
* fasttext (compiled from source code)

# How to use?

There is a jupyter notebook describing all the steps for executing the C² method. Some parts need to be adapted to your hardware requirements (if you have multiple GPUs).

The jupyter notebook is available here: [meli2019.ipynb](meli2019.ipynb).

# License

This software is available under [MIT license](LICENSE.MIT).
