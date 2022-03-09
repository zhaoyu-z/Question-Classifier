import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.autograd import Variable
from global_parser import parser
from glove import read_glove
from vocabulary import Vocabulary
from split_label import SplitLabel

class BagOfWords(nn.Module):

    def __init__(self, tagset_size, word_embeddings, file_path):
        self.embedding_dim = int(parser['Network Structure']['word_embedding_dim'])
        self.hidden_dim = int(self.embedding_dim*2/3)
        # self.hidden_dim = 128

        super(BagOfWords, self).__init__()

        self.word_embeddings = word_embeddings

        self.hidden2tag = nn.Linear(self.embedding_dim, self.hidden_dim)

        self.activation_function1 = nn.Tanh()

        self.hidden3tag = nn.Linear(self.hidden_dim, tagset_size)

    def forward(self, sentence):
        sentence_vector = self.word_embeddings(sentence)
        sentence_vector = torch.mean(sentence_vector, dim=0)
        tag_space = self.hidden2tag(sentence_vector)

        tag_space = self.activation_function1(tag_space)

        tag_space = self.hidden3tag(tag_space)

        tag_scores = F.log_softmax(tag_space, dim = -1)

        return tag_scores.unsqueeze(0)

def train(file_path):
    pretrained = parser['Options for model']['pretrained']
    freeze = eval(parser['Options for model']['freeze'])
    train_path = parser['Paths To Datasets And Evaluation']['path_train']
    word_dim = parser['Network Structure']['word_embedding_dim']
    learning_rate = float(parser['Hyperparameters']['lr_param'])
    model_type = parser['Options for model']['model']
    if eval(pretrained):
        if freeze:
            print("Training pretrained and freeze", model_type, "model...")
        else:
            print("Training pretrained and fine-tuning", model_type, "model...")
        print()
        glove_path = parser['Using pre-trained Embeddings']['path_pre_emb']
        vec = read_glove(glove_path)
        voca = Vocabulary("train", word_dim)
        voca.set_word_vector(vec)
        voca.from_word2vect_word2ind()
        voca.from_word2vect_wordEmbeddings(freeze)
        word_emb = voca.get_word_embeddings()
    else:
        print("Training random", model_type, "model...")
        print()
        voca = Vocabulary("train", word_dim)
        voca.setup('../data/train.txt')
        word_emb = voca.get_word_embeddings()
    features,labels = SplitLabel(train_path).generate_sentences()

    tag_to_ix = {}
    id = 0
    for l in labels:
        if l not in tag_to_ix.keys():
            tag_to_ix[l] = id
            id += 1

    model = BagOfWords(len(tag_to_ix), word_emb, file_path)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = optim.Adam(model.parameters(), lr=0.05)
    #optimizer = torch.optim.Adam(model.parameters(), lr= 0.01)

    for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in zip(features, labels):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = voca.get_sentence_ind(sentence,"Bow")
            targets = torch.tensor([tag_to_ix[tags]],dtype=torch.long)

            # Step 3. Run our forward pass.
            tag_scores = model.forward(sentence_in)
            #
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)

            loss.backward()
            optimizer.step()

        print("Epoch", epoch)
        with torch.no_grad():
            y_true = []
            y_pred = []
            most_error_label = {}
            for sentence, tags in zip(features,labels):
                sentence_in = voca.get_sentence_ind(sentence,"Bow")
                tag_scores = model.forward(sentence_in)

                ind = torch.argmax(tag_scores)

                y_pred.append(ind)
                y_true.append(tag_to_ix[tags])

                if tag_to_ix[tags] != ind:
                    if tags not in most_error_label.keys():
                        most_error_label[tags] = 1
                    else:
                        most_error_label[tags] += 1

            most_error_label = sorted(most_error_label.items(), key=lambda item: item[1], reverse=True)[:3]

            print("Accuracy", accuracy_score(y_true,y_pred))
            print("F1-score", f1_score(y_true,y_pred,average='macro'))
            # print("Confusion_matrix \n", confusion_matrix(y_true,y_pred))
            # print("First three most frequent misclassifed label:",most_error_label)
            print()

    torch.save(model, parser['Options for model']['model_save_path'])

    tag2index_filepath = parser['Options for model']['tag2index_save_path']
    tag2index_file = open(tag2index_filepath, "wb")
    pickle.dump(tag_to_ix, tag2index_file)
    tag2index_file.close()

    voca_filepath = parser['Options for model']['voca_save_path']
    voca_file = open(voca_filepath, "wb")
    pickle.dump(voca, voca_file)
    voca_file.close()
    # return model, tag_to_ix

def test(file_path,model=None, tag_to_ix=None):
    pretrained = eval(parser['Options for model']['pretrained'])
    model_type = parser['Options for model']['model']
    if pretrained:
        freeze = eval(parser['Options for model']['freeze'])
        if freeze:
            print("Testing pretrained and freeze", model_type, "model...")
        else:
            print("Testing pretrained and fine-tuning", model_type, "model...")
    else:
        print("Testing random", model_type, "model...")
    print()

    model_path = parser['Options for model']['model_save_path']
    model = torch.load(model_path)
    model.to('cpu')
    tag2index_path = parser['Options for model']['tag2index_save_path']
    tag2index_file = open(tag2index_path, "rb")
    tag_to_ix = pickle.load(tag2index_file)

    voca_filepath = parser['Options for model']['voca_save_path']
    voca_file = open(voca_filepath, "rb")
    voca = pickle.load(voca_file)

    test_path = parser['Paths To Datasets And Evaluation']['path_test']
    features, labels = SplitLabel(test_path).generate_sentences()

    with torch.no_grad():
        y_true = []
        y_pred = []
        most_error_label = {}
        for sentence, tags in zip(features,labels):
            sentence_in = voca.get_sentence_ind(sentence,"Bow")
            tag_scores = model.forward(sentence_in)

            ind = torch.argmax(tag_scores)

            y_pred.append(ind)
            y_true.append(tag_to_ix[tags])

            if tag_to_ix[tags] != ind:
                if tags not in most_error_label.keys():
                    most_error_label[tags] = 1
                else:
                    most_error_label[tags] += 1

        most_error_label = sorted(most_error_label.items(), key=lambda item: item[1], reverse=True)[:3]

        print("Accuracy", accuracy_score(y_true,y_pred))
        print("F1-score", f1_score(y_true,y_pred,average='macro'))
        print("Confusion_matrix \n", confusion_matrix(y_true,y_pred))
        print("First three most frequent misclassifed label:",most_error_label)
