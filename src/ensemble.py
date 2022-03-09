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

def test(file_path,model=None, tag_to_ix=None):

    print("Testing Ensemble Stored trained bilstm and bow model")

    output_file = parser['Evaluation']['path_eval_result']

    model_path_bow = parser['Options for model']['model_save_path_bow']
    model_path_bilstm = parser['Options for model']['model_save_path_bilstm']
    model_bow = torch.load(model_path_bow)
    model_bilstm = torch.load(model_path_bilstm)

    model_bow.to('cpu')
    model_bilstm.to('cpu')

    tag2index_path_bow = parser['Options for model']['tag2index_save_path_bow']
    tag2index_file_bow = open(tag2index_path_bow, "rb")
    tag_to_ix_bow = pickle.load(tag2index_file_bow)

    tag2index_path_bilstm = parser['Options for model']['tag2index_save_path_bilstm']
    tag2index_file_bilstm = open(tag2index_path_bilstm, "rb")
    tag_to_ix_bilstm = pickle.load(tag2index_file_bilstm)

    voca_filepath_bow = parser['Options for model']['voca_save_path_bow']
    voca_file_bow = open(voca_filepath_bow, "rb")
    voca_bow = pickle.load(voca_file_bow)

    voca_filepath_bilstm = parser['Options for model']['voca_save_path_bilstm']
    voca_file_bilstm = open(voca_filepath_bilstm, "rb")
    voca_bilstm = pickle.load(voca_file_bilstm)

    test_path = parser['Paths To Datasets And Evaluation']['path_test']
    features, labels = SplitLabel(test_path).generate_sentences()

    with torch.no_grad():
        y_true = []
        y_pred = []
        most_error_label = {}
        with open(output_file, 'w') as f:
            for sentence, tags in zip(features,labels):
                sentence_in_bow = voca_bow.get_sentence_ind(sentence,"Bow")
                tag_scores_bow = model_bow.forward(sentence_in_bow)

                sentence_in_bilstm = voca_bilstm.get_sentence_ind(sentence,"Bilstm")
                tag_scores_bilstm = model_bilstm.forward(sentence_in_bilstm)

                tag_scores = tag_scores_bow + tag_scores_bilstm
                ind = torch.argmax(tag_scores)

                predicted_label = list(tag_to_ix_bow.keys())[list(tag_to_ix_bow.values()).index(ind)]
                f.write(str(predicted_label) + "\n")
                y_pred.append(ind)
                y_true.append(tag_to_ix_bow[tags])

                if tag_to_ix_bow[tags] != ind:
                    if tags not in most_error_label.keys():
                        most_error_label[tags] = 1
                    else:
                        most_error_label[tags] += 1

            most_error_label = sorted(most_error_label.items(), key=lambda item: item[1], reverse=True)[:3]
            accuracy = accuracy_score(y_true,y_pred)
            f.write("Overall Accuracy of Ensemble stored models:" + str(accuracy))
            f.close()

        print("Accuracy", accuracy)
        print("F1-score", f1_score(y_true,y_pred,average='macro'))
        print("Confusion_matrix \n", confusion_matrix(y_true,y_pred))
        print("First three most frequent misclassifed label:",most_error_label)
