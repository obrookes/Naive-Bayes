import math

from nltk import FreqDist
from nltk.corpus import inaugural


class Data:

    def __init__(self, documents, cls):
        """ Initialise data class and train model"""
        self._documents = documents
        self._cls = cls

        self.set_vocab()  # A list of words said *without* repition
        self.set_log_prior()
        self.set_big_doc()  # dictionary, value is a list of words with rep
        self.set_log_liklihood()

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
        print(self._log_prior)

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

    def display_vocab(self, index):
        print(self.vocab[:index])

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

    def display_big_doc(self, index):
        for c in self._cls:
            print(c, self._big_doc[c][:index])


    def set_log_liklihood(self):
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

def tokenize(string):
    string = string.lower()
    return string.split(' ')

def run():

    # Dataset generation
    cls = ['Obama', 'Trump']
    obama_sentences = inaugural.sents('2009-Obama.txt')
    trump_sentences = inaugural.sents('2017-Trump.txt')
    labelled_obama = [(s, cls[0]) for s in obama_sentences]
    labelled_trump = [(s, cls[1]) for s in trump_sentences]
    labelled_data = labelled_obama + labelled_trump

    foo = Data(labelled_data, cls)

    trump_test = [
            'We', ',', 'the', 'citizens', 'of', 'America', ',', 'are', 'now',
            'joined', 'in', 'a', 'great', 'national', 'effort', 'to',
            'rebuild', 'our', 'country', 'and', 'restore', 'its', 'promise',
            'for', 'all', 'of', 'our', 'people', '.']

    obama_test = [
            'I', 'stand', 'here', 'today', 'humbled', 'by', 'the',
            'task', 'before', 'us', ',', 'grateful', 'for', 'the', 'trust',
            'you', 'have', 'bestowed', ',', 'mindful', 'of', 'the',
            'sacrifices', 'borne', 'by', 'our', 'ancestors', '.']

    print("trump test")
    print(foo.test_doc(trump_test))

    print("obama test")
    print(foo.test_doc(obama_test))

    print('trying to be trump')
    print(foo.test_doc(tokenize('We are humbled sacrafice')))

if __name__ == '__main__':
    run()
