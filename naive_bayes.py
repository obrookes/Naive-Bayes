import math

from nltk import FreqDist
from nltk.corpus import inaugural


class Data:

    def __init__(self, documents, cls):
        self._documents = documents
        self._cls = cls

        self.set_vocab()  # A list of words said *without* repition
        self.set_log_prior()
        self.set_big_doc()  # dictionary, value is a list of words with rep
        self.set_log_liklihood()

    def set_log_prior(self):
        self.log_prior = {}  # The key will be one of the labels
        n_doc = len(self._documents)
        for c in self._cls:
            n_c = len([doc for doc in self._documents if doc[1] == c])
            prior = n_c / n_doc
            self.log_prior[c] = math.log(prior, 2)

    def set_vocab(self):
        all_words = []
        for pair in self._documents:
            document = pair[0]
            for word in document:
                all_words.append(word)
        self.vocab = set(all_words)

    def set_big_doc(self):
        self._big_doc = {}
        for c in self._cls:
            self._big_doc[c] = []
            for doc, label in self._documents:
                if label == c:
                    self._big_doc[c] += doc

    def set_log_liklihood(self):
        self.log_likelihood = {}
        for c in self._cls:
            self.log_likelihood[c] = {}
            denom = 0
            fd = dict(FreqDist(w.lower() for w in self._big_doc[c]))

            for word in self.vocab:
                denom += (self._big_doc[c].count(word) + 1)

            for word in self.vocab:
                try:
                    count = fd[word] + 1
                except KeyError:
                    count = 1
                log_likelihood = math.log((count / denom), 2)
                self.log_likelihood[c][word] = log_likelihood

    def test_doc(self, doc):  # doc is a list of strings
        sm = {}
        for c in self._cls:
            sm[c] = self.log_prior[c]
            for word in doc:
                if word in self.vocab:
                    sm[c] += self.log_likelihood[c][word]
        return(max(sm, key=sm.get))


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


if __name__ == '__main__':
    run()
