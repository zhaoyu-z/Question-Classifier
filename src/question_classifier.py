import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import string
import random
from global_parser import parser as conf_parser
import argparse
import bilstm
import bow

def train(confi_file_path):
    torch.manual_seed(1)
    random.seed(1)

    conf_parser.read(confi_file_path)

    if confi_file_path.split('/')[-1] == 'bilstm.config':
        bilstm.train(confi_file_path)
    else:
        bow.train(confi_file_path)

def test(confi_file_path):
    torch.manual_seed(1)

    random.seed(1)

    conf_parser.read(confi_file_path)

    if confi_file_path.split('/')[-1] == 'bilstm.config':
        bilstm.test(confi_file_path)
    else:
        bow.test(confi_file_path)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Configuration file')
parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
args = parser.parse_args()
if args.train:
   train(args.config)
elif args.test:
   test(args.config)
