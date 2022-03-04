import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import string
import random
import configparser
import argparse
import bilstm

class BagOfWords:

    def __init__(self,sentences,volcab,word_embedding_dim):
        self.sentences = sentences
        self.volcab = volcab
        self.word_embedding_dim = word_embedding_dim

    def words(self,sentence):
        words = sentence.split()
        words = list(
            filter(lambda token: token not in string.punctuation and token not in ["``","''"], words))
        lower_words =[w.lower() for w in words]
        return lower_words

    def bow_vec(self):
        bow_vecs = []
        for sentence in self.sentences:
            sentence_word = self.words(sentence)
            bow_vector = numpy.zeros(self.word_embedding_dim)
            for w in sentence_word:
                bow_vector += self.volcab.get(w)
            bow_vector[:] = [x / len(sentence_word) for x in bow_vector]
            bow_vecs.append(bow_vector)
        return bow_vecs

#"../data/config.ini"
def train(confi_file_path):
    torch.manual_seed(1)

    random.seed(1)

    config = configparser.ConfigParser()

    config.sections()

    config.read(confi_file_path)

    if confi_file_path.split('/')[-1] == 'bilstm.config':
        bilstm.train()
    # else:
    #     bow.train()

def test(confi_file_path):
    torch.manual_seed(1)

    random.seed(1)

    config = configparser.ConfigParser()

    config.sections()

    config.read(confi_file_path)

    if confi_file_path.split('/')[-1] == 'bilstm.config':
        bilstm.test()
    # else:
    #     bow.test()


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Configuration file')
parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
args = parser.parse_args()
if args.train:
   train(args.config)
elif args.test:
   test(args.config)
