import numpy as np
import nltk
import math
import os

from nltk import FreqDist

class Data:

    def __init__(self):
        self._logprior = {}
        # logliklihood for each word for each class

    def set_log_prior(self, training_set, c):
        label = 1
        n_doc = len(training_set)
        n_c = len([data for data in labelled_data if t[label] == c])
        prior = n_c / n_doc
        self._logprior[c] = math.log(prior, 2)

    def set_big_v():
        obama_words = inaugural.words('2009-Obama.txt')
        trump_words = inaugural.words('2017-Trump.txt')
        all_words = obama_words + trump_words
        self._big_v = set(all_words)

# Dataset

if __name__ == "__maain__":

    # print(os.listdir(nltk.data.find('corpora')))
    from nltk.corpus import inaugural
    # print(inaugural.fileids())
    # print(inaugural.raw('2009-Obama.txt'))

    # list of Obama sentences
    obama_sentences = inaugural.sents('2009-Obama.txt')
    trump_sentences = inaugural.sents('2017-Trump.txt')

    labels = ['Obama', 'Trump']

    labelled_obama = [(s, labels[0]) for s in obama_sentences]
    labelled_trump = [(s, labels[1]) for s in trump_sentences]
    labelled_data = labelled_obama + labelled_trump

    # calculate class priors

    trump_prior = len([t for t in labelled_data if t[1] == 'Trump']) / len(labelled_data)
    obama_prior = len([t for t in labelled_data if t[1] == 'Obama']) / len(labelled_data)
    # print(trump_prior, obama_prior)

    # compute liklihoods

    def log_prior(training_set, c):

        label = 1

        n_doc = len(training_set)
        n_c = len([data for data in labelled_data if t[label] == c])
        prior = n_c / n_doc
        return math.log(prior, 2)

    def get_vocab():
        obama_words = inaugural.words('2009-Obama.txt')
        trump_words = inaugural.words('2017-Trump.txt')
        all_words = obama_words + trump_words
        return set(all_words)

    def big_doc(c):
        if c == 'Obama':
            sents = inaugural.sents('2009-Obama.txt')
        elif c == 'Trump':
            sents = inaugural.sents('2017-Trump.txt')

        return [w for s in sents for w in s]

    # for trump sents (flattened)

    big_doc_trump = big_doc('Trump')
    denom = 0
    fd = dict(FreqDist(w.lower() for w in big_doc_trump))

    big_v = get_vocab()

    for word in big_v:
        denom += (big_doc_trump.count(word) + 1)

    trump_loglikihood = [] # list of word: liklihoods

    for word in big_v:
        try:
            count = fd[word] + 1
        except KeyError:
            count = 1
        log_likihood = math.log((count / denom), 2)
        trump_loglikihood.append((word, log_likihood))



    # for obama sents (flattened)

    big_doc_obama = big_doc('Obama')
    denom = 0
    fd = dict(FreqDist(w.lower() for w in big_doc_obama))

    for word in big_v:
        denom += (big_doc_obama.count(word) + 1)


    obama_loglikihood = [] # list of word: liklihoods

    for word in big_v:
        try:
            count = fd[word] + 1
        except KeyError:
            count = 1
        log_likihood = math.log((count / denom), 2)
        obama_loglikihood.append((word, log_likihood))

    





















    # from nltk import FreqDist
    # fd_obama_words = FreqDist(w.lower() for w in obama_words)





# class_probability = documents_of_class / total_number_of_documents
# liklihood_functuion = count(w, c) + 1 / count(c) + mod(vocabulary)
