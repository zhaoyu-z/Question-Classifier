import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy
import string
import random

import configparser

import argparse

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


class BiLSTMTagger(nn.Module):

   def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
       super(BiLSTMTagger, self).__init__()
       self.hidden_dim = hidden_dim
       self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

       self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

       self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

   def forward(self, sentence):
        embeds = self.word_embeddings(sentence)

        bilstm_out, (h_n, c_n) = self.lstm(embeds.view(len(sentence), 1, -1))

        tag_space = self.hidden2tag(bilstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


#"../data/config.ini"
def train(confi_file_path):
    torch.manual_seed(1)

    random.seed(1)

    config = configparser.ConfigParser()

    config.sections()

    config.read(confi_file_path)

    #overall5500 = config["Paths To Datasets And Evaluation"]["path_overall"]


def test(confi_file_path):
    torch.manual_seed(1)

    random.seed(1)

    config = configparser.ConfigParser()

    config.sections()

    config.read(confi_file_path)

    #overall5500 = config["Paths To Datasets And Evaluation"]["path_overall"]


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Configuration file')
parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
args = parser.parse_args()
if args.train:
   train(args.config)
elif args.test:
   test(args.config)
