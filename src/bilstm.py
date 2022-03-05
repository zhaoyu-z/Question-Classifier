import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing import parser
import pickle
from preprocessing import get_embvec
from glove import read_glove
from split_label import SplitLabel
from vocabulary import Vocabulary
class BiLSTMTagger(nn.Module):

   def __init__(self, tagset_size, vol,file_path):
       self.embedding_dim = int(parser['Network Structure']['word_embedding_dim'])
       self.batch_size = int(parser['Network Structure']['batch_size'])

       super(BiLSTMTagger, self).__init__()
       self.hidden_dim = int(parser['Network Structure']['hidden_dim'])

       self.vol = vol

       self.bilstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True)

       self.hidden2tag = nn.Linear(self.hidden_dim * 2, tagset_size)


   def bilstm_vec(self, sentence):
       pretrained = eval(parser['Options for model']['pretrained'])
       freeze = eval(parser['Options for model']['freeze'])
       embeds = get_embvec(sentence, self.vol, self.embedding_dim) # nn.Embedding.from_pretrained object

       if pretrained:
           #print(freeze)
           embeds = nn.Embedding.from_pretrained(embeds, freeze=freeze)
           # print(embeds.weight.requires_grad)
           # bilstm_out, (h_n, c_n) = self.bilstm(embeds.view(len(embeds), 1, -1))
           embeds_num = embeds.weight.data.numpy()
           embeds_num = embeds_num.reshape(len(embeds_num), 1, -1)
           #bilstm_out, (h_n, c_n) = self.bilstm(embeds.view(len(embeds_num), 1, -1))
           bilstm_out, (h_n, c_n) = self.bilstm(torch.from_numpy(embeds_num))
       else:
           bilstm_out, (h_n, c_n) = self.bilstm(embeds.view(len(embeds), 1, -1))

       out = torch.hstack((h_n[-2, :, :], h_n[-1, :, :]))

       return out


   def forward(self, sentence):
        bilstm_out = self.bilstm_vec(sentence)

        tag_space = self.hidden2tag(bilstm_out)

        tag_scores = F.log_softmax(tag_space, dim = -1)

        return tag_scores

def train(file_path):
    pretrained = parser['Options for model']['pretrained']
    train_path = parser['Paths To Datasets And Evaluation']['path_train']
    word_dim = parser['Network Structure']['word_embedding_dim']
    if eval(pretrained):
        print("pretrained")
        glove_path = parser['Using pre-trained Embeddings']['path_pre_emb']
        vec = read_glove(glove_path)
    else:
        print("random")
        voca = Vocabulary("train", word_dim)
        voca.setup('../data/train.txt')
        vec = voca.get_word2vector()
    features,labels = SplitLabel(train_path).generate_sentences()

    tag_to_ix = {}
    id = 0
    for l in labels:
        if l not in tag_to_ix.keys():
            tag_to_ix[l] = id
            id += 1

    model = BiLSTMTagger(len(tag_to_ix), vec, file_path)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    #optimizer = optim.Adam(model.parameters(), lr=0.05)
    #optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)

    for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in zip(features, labels):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            targets = torch.tensor([tag_to_ix[tags]],dtype=torch.long)

            # Step 3. Run our forward pass.
            tag_scores = model.forward(sentence)
            #
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)

            loss.backward()
            optimizer.step()

        print("Epoch", epoch)
        with torch.no_grad():
            acc = 0
            for sentence, tags in zip(features,labels):

                tag_scores = model.forward(sentence)

                ind = torch.argmax(tag_scores)

                k = [k for k, v in tag_to_ix.items() if v == ind]

                if k[0] == tags:
                    acc += 1
            print(acc / len(features))

    torch.save(model, parser['Options for model']['model_save_path'])

    tag2index_filepath = parser['Options for model']['tag2index_save_path']
    tag2index_file = open(tag2index_filepath, "wb")
    pickle.dump(tag_to_ix, tag2index_file)
    tag2index_file.close()
    # return model, tag_to_ix

def test(file_path,model=None, tag_to_ix=None):
    model_path = parser['Options for model']['model_save_path']
    model = torch.load(model_path)
    model.to('cpu')
    tag2index_path = parser['Options for model']['tag2index_save_path']
    tag2index_file = open(tag2index_path, "rb")
    tag_to_ix = pickle.load(tag2index_file)

    test_path = parser['Paths To Datasets And Evaluation']['path_test']
    features, labels = SplitLabel(test_path).generate_sentences()

    with torch.no_grad():
        acc = 0
        for sentence, tags in zip(features, labels):

            tag_scores = model.forward(sentence)

            ind = torch.argmax(tag_scores)

            k = [k for k, v in tag_to_ix.items() if v == ind]

            if k[0] == tags:
                acc += 1
        print(acc / len(features))

def test_one(model=None, tag_to_ix=None, test_sentence="How are you ?"):
    model_path = parser['Options for model']['model_save_path']
    model = torch.load(model_path)
    model.to('cpu')
    tag2index_path = parser['Options for model']['tag2index_save_path']
    tag2index_file = open(tag2index_path, "rb")
    tag_to_ix = pickle.load(tag2index_file)

    pretrained = parser['Options for model']['pretrained']
    if pretrained:
        glove_path = parser['Using pre-trained Embeddings']['path_pre_emb']
        glove_vec = read_glove(glove_path)
        embedding_dim = int(parser['Network Structure']['word_embedding_dim'])

        with torch.no_grad():
            inputs = get_embvec(test_sentence, glove_vec, embedding_dim)
            # print(inputs)
            tag_scores = model.forward(test_sentence)
            # print(tag_scores)

            ind = torch.argmax(tag_scores)

            k = [k for k, v in tag_to_ix.items() if v == ind]

            print(k)
# train()
# test()
