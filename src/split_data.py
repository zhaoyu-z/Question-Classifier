import random
import numpy as np
import configparser

def split_train_dev_data(data_file_path, train_save_path, dev_save_path):

    file = open(data_file_path)
    lines = file.readlines()
    file.close()

    ## 90:10 ratio, 90% train, 10% development
    np.random.shuffle(lines)
    train_data, dev_data = np.split(lines, [int(0.9 * len(lines))])

    # write the train data into train.txt
    file_train = open(train_save_path, 'w')
    for i in range(len(train_data)):
        file_train.write(train_data[i])
    file_train.close()

    # write the dev data into dev.txt
    file_dev = open(dev_save_path, 'w')
    for i in range(len(dev_data)):
        file_dev.write(dev_data[i])
    file_dev.close()


def split(config_file_path):

    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    data_file_path = config['Paths To Datasets And Evaluation']['path_overall']
    train_save_path = config['Paths To Datasets And Evaluation']['path_train']
    dev_save_path = config['Paths To Datasets And Evaluation']['path_dev']

    split_train_dev_data(data_file_path, train_save_path, dev_save_path)

if __name__ == '__main__':
    split('../data/bow.config')
