from nltk.corpus import inaugural
from naive_bayes import Data

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
