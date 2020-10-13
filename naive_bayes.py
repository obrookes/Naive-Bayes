import math
from nltk import FreqDist

class Model:

    def __init__(self, documents, cls):
        """ Initialise data class and train model"""
        self._documents = documents
        self._cls = cls

        self.set_vocab()  # A list of words said *without* repition
        self.set_log_prior()
        self.set_big_doc()  # dictionary, value is a list of words with rep
        self.set_log_likelihood()

    def set_log_prior(self):
        """ calculate log priors: 
            - n_doc = total number of documents
            - n_c = number of documents of particular class
            - prior = n_c / n_doc 
            """
        self.log_prior = {}  # The key will be one of the labels
        n_doc = len(self._documents)
        for c in self._cls:
            n_c = len([doc for doc in self._documents if doc[1] == c])
            prior = n_c / n_doc
            self.log_prior[c] = math.log(prior, 2)

    def display_log_prior(self):
        print(self.log_prior)

    def set_vocab(self):
        """ words from doucments without repition:
            - pair is a (document, label) tuple
            - document is an array of strings
            - set() function removes duplicates
            """
        all_words = []
        for pair in self._documents:
            document = pair[0]
            for word in document:
                all_words.append(word)
        self.vocab = set(all_words)

    def display_vocab(self, index=10):
        print(list(self.vocab)[:index])

    def set_big_doc(self):
        """ all words in class with repitition:
            - iterate through training set
            - extract all word occurences for each class
            """
        self._big_doc = {}
        for c in self._cls:
            self._big_doc[c] = []
            for doc, label in self._documents:
                if label == c:
                    self._big_doc[c] += doc

    def display_big_doc(self, index=10):
        for c in self._cls:
            print(c, self._big_doc[c][:index])


    def set_log_likelihood(self):
        """ calculate log likelihoods:
            - unique_class_words: number of unique words the class
            - bag_of_words: frequence distribution of all words in the class
            - try except: in case a word is not used by the class
            """
        self.log_likelihood = {}
        for c in self._cls:
            class_log_likelihood = {}

            # denominator
            unique_class_words = 0
            for word in self.vocab:
                unique_class_words += (self._big_doc[c].count(word) + 1)

            bag_of_words = dict(FreqDist(w.lower() for w in self._big_doc[c]))
            for word in self.vocab:
                # numerator
                try:
                    count = bag_of_words[word] + 1
                except KeyError:
                    count = 1
                class_log_likelihood[word] = math.log((count / unique_class_words), 2)

            self.log_likelihood[c] = class_log_likelihood

    def display_log_likelihood(self, index=10):
        for c in self._cls:
            print(c, list(self.log_likelihood[c].items())[:index])


    def test_doc(self, doc):
        """ classify document:
            - doc: tokenised string (document)
            - assigns a log likelihood to each each token (word)
            - sums over likelihoods 
            - maximises over classes to give final classification
        """
        sm = {}
        for c in self._cls:
            sm[c] = self.log_prior[c]
            for word in doc:
                if word in self.vocab:
                    sm[c] += self.log_likelihood[c][word]
        return(max(sm, key=sm.get))
