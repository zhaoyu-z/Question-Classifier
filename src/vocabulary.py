import numpy as np
import string
from split_label import SplitLabel
import configparser
class Vocabulary:
    UNK_token = 0
    def __init__(self, name, dim):
        self.name = name
        self.word2ind = {}
        self.word2count = {}

        self.word2vec = {"#UNK#":np.random.normal(0, 1, dim)}
        self.ind2word = ["#UNK#"]
        self.num_words = 1
        self.stopwords = []
        self.dim = dim

    def read_stop_words(self, path):
        with open(path) as file:
            words = file.read().split("\n")
            words = list(filter(None, words))
            self.stopwords  = words


    def add_word(self, word):
        word = word.lower()
        if word not in self.word2count  and word not in string.punctuation and word not in ["``","''"] and word not in self.stopwords:
            self.word2count[word] = 1
            self.word2vec[word] = np.zeros(self.dim)
        elif word in self.word2count:
            self.word2count[word] = self.word2count[word] + 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def filter(self, threshold):
        self.word2count = {k: v for k, v in self.word2count.items() if v >= threshold}
        other_words = self.word2count.keys()
        self.ind2word += other_words
        for i in range(len(self.ind2word)):
            self.word2ind[self.ind2word[i]]= i
        self.num_words = len(self.ind2word)



corpus = SplitLabel("../data/train.txt")
features,_ = corpus.generate_sentences()
#print(features)


voca = Vocabulary("train", 300)
voca.read_stop_words('../data/stopwords.txt')
for s in features:
    voca.add_sentence(s)
print(len(voca.word2count.keys()))

parser = configparser.ConfigParser()
parser.sections()
parser.read("../data/bow.config")

voca.filter(int(parser['Hyperparameters']['vocab_threshold']))
print(dict(sorted(voca.word2count.items(), key=lambda item: item[1])))
print(max(voca.word2count.values()))
print(voca.num_words)



# with open("../data/NLTK's list of english stopwords", 'r') as file:
#     words = file.read().split("\n")
#     words = list(filter(None, words))
#     print(words)
