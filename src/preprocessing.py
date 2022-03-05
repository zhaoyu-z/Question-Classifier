import string
import torch
import torch.nn as nn
import configparser

parser = configparser.ConfigParser()
parser.sections()

def get_embvec(sentence, vol, embedding_dim):
    model_type = parser['Options for model']['model']
    words = sentence.split()
    if model_type == 'bow':
        words = list(filter(lambda token: token not in string.punctuation, words))
    vec = torch.FloatTensor(len(words), embedding_dim)
    for i in range(0, len(words)):
        if words[i] in vol.keys():
            vec[i] = (vol.get(words[i]))
        else:
            vec[i] = (vol.get("#UNK#"))

    return vec
