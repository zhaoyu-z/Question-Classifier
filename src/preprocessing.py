import string
import torch

def get_embvec(sentence, vol, embedding_dim):
    words = sentence.split()
    words = list(filter(lambda token: token not in string.punctuation, words))
    vec = torch.FloatTensor(len(words), embedding_dim)
    for i in range(0, len(words)):
        if words[i] in vol.keys():
            vec[i] = (vol.get(words[i]))
        else:
            vec[i] = (vol.get("#UNK#"))

    return vec
