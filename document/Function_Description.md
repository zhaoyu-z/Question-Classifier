### Here is the function description file.

#### bilstm.py

```python
class BiLSTMTagger(nn.Module)
"""
A class used to generate the Bilstm model
...
Attributes
----------
tagset_size: int
    the size of taget set, i.e., the number of target predicted labels
file_path: str
    the file path of configuration file
word_embeddings: nn
    convert to word embeddings vectors function
self.embedding_dim: int
    the dimension of word embeddings vectors
self.batch_size: int
    the size of the batch
self.hidden_dim: int
    the dimension of hidden layer
self.bilstm: nn
    the nn.LSTM model set in bi-directional
self.word_embeddings: nn
    convert to word embeddings vectors function
self.hidden2tag: nn
    hidden-output layer setting

"""
"""
Methods
----------
"""
def forward(self, sentence_in)
"""
    The forward step of the bilstm model in neural network by predicting the 
    target label of the input sentence_in
    ...
    Attributes
    ----------
    sentence_in: str
        The input sentence used to update neural network

    Returns
    ----------
    tag_scores: torch
        The evaluated scores for each known target label in training dataset
"""

def train(file_path)
"""
Training step of the bilstm model

Read from the input configuration file and generate the sentence word embedding
to train the classifier according to the parameters, and update the classifier
after each time training, then store the trained model for testing
...
Attributes
----------
file_path: str
    the file path of configuration file

"""

def test(file_path,model=None, tag_to_ix=None)
"""
Test the trained bilstm model from train(file_path) function

Load the trained model and test on the testing dataset to evaluate the
performance of the trained model on testing dataset
...
Attributes
----------
file_path: str
    the file path of configuration file
model: BiLSTMTagger
    the stored trained Bilstm model
tag_to_ix: {str: int}
    the taget dictionary with its index
"""
```

#### bow.py

```python
class BagOfWords(nn.Module)
"""
A class used to generate the Bag of Words(Bow) model
...
Attributes
----------
tagset_size : int
    the size of taget set, i.e., the number of target predicted labels
word_embeddings: nn
    convert to word embeddings vectors function
file_path: str
    the file path of configuration file
self.embedding_dim: int
    the word embedding dimension
self.hidden_dim: int
    the hidden layer dimension
self.word_embeddings: nn
    convert to word embeddings vectors function
self.hidden2tag: nn
    input-hidden layer setting
self.activation_function1: nn
    Tanh function for activation function of hidden layer
self.hidden3tag: nn
    hidden-output layer setting 
"""
"""
Methods
----------
"""
def forward(self, sentence_in)
"""
    The forward step of the Bow model in neural network by predicting the 
    target label of the input sentence_in
    ...
    Attributes
    ----------
    sentence_in: str
        The input sentence used to update neural network

    Returns
    ----------
    tag_scores: torch
        The evaluated scores for each known target label in training dataset
"""

def train(file_path)
"""
Training step of the bow model

Read from the input configuration file and generate the sentence word embedding
to train the classifier according to the parameters, and update the classifier
after each time training, then store the trained model for testing
...
Attributes
----------
file_path: str
    the file path of configuration file

"""

def test(file_path,model=None, tag_to_ix=None)
"""
Test the trained bow model from train(file_path) function

Load the trained model and test on the testing dataset to evaluate the
performance of the trained model on testing dataset
...
Attributes
----------
file_path: str
    the file path of configuration file
model: BagOfWords
    the stored trained Bow model
tag_to_ix: {str: int}
    the taget dictionary with its index
"""
```

#### ensemble.py

```python
def test(file_path,model=None, tag_to_ix=None)
"""
Test the ensembled trained bow model and bilstm model

Load the trained models and ensemble them to test on the testing dataset to 
evaluate the performance of the ensembled model
...
Attributes
----------
file_path: str
    the file path of configuration file
model: BagOfWords
    the stored trained Bow model
tag_to_ix: {str: int}
    the taget dictionary with its index
"""
```

#### global_parser.py

```python
"""
The interface for classifiers implemented by configparser library
"""
```

#### glove.py

```python
def read_by_tokens(fileobj)
"""
Read the file and generate the tokens
...
Attributes
----------
fileobj: file object
    The file read
"""

def read_glove(path)
"""
Read the glove.small.txt file and generate the dictionary of word and word 
embedding vectors.
...
Attributes
----------
path: str
    the glove.small.txt file path

Returns
----------
glove_vec: {str: torch}
    The dictionary of word and word embedding vectors from glove

"""
```

#### question_classifier.py

```python
def train(confi_file_path)
"""
Training step for models
...
Attributes
----------
confi_file_path: str
    The path of configuration file
"""

def test(confi_file_path)
"""
Testing step for models
...
Attributes
----------
confi_file_path: str
    The path of configuration file
"""

"""
The file is the main program by reading different configuration files to train 
or test the bilstm, bow or ensemble classifiers.
"""
```

#### split_data.py

```python
def split_train_dev_data(data_file_path, train_save_path, dev_save_path)
"""
Split the input dataset into training dataset and developing dataset and store
them into different files
...
Attributes
----------
data_file_path: str
    The input dataset file path
train_save_path: str
    The stored file path of training dataset
dev_save_path: str
    The stored file path of developing dataset
"""

def split(config_file_path)
"""
Split the original dataset into training and developing version
...
Attributes
----------
config_file_path: str
    The input file path of the original dataset
"""
```

#### split_label.py

```python
class SplitLabel
"""
A class used to split the input dataset into questions and labels
...
Attributes
----------
corpus_path: str
    The file path of the input dataset
"""
"""
Methods
----------
"""
def generate_sentences(self)
"""
    The method is used to split the input file with file path corpus_path
    into questions and labels
    ...
    Returns
    ----------
    features, labels: str, str
        features are the questions and labels are the corresponding labels

"""
```

#### vocabulary.py

```python
class Vocabulary:
"""
A class used to generate the word embeddings
...
Attributes
----------
name: str
    The name of the classifier
dim: int
    The dimension of word embeddings
self.name: str
    The name of the classifier
self.word2ind: {str: int}
    The dictionary of index which corresponds to words
self.word2count: {str: int}
    The dictionary of counter used to count the number of times word in file
self.word_embeddings: nn
    The word_embeddings function
self.word2vec: {str: torch}
    The dictionary of word embeddings vectors corresponding to words
self.ind2word: [str]
    The list of words used to get the corresponding word with its index
self.num_words: int
    The counter of words
self.stopwords: [str]
    The array used to store stopwords
self.dim: int
    The dimension of word embeddings

"""
"""
Methods
----------
"""
def read_stop_words(self, path)
"""
    The method is used to read the stop words
    ...
    Attributes
    ----------
    path: str
        The file path of the stop words file
"""

def add_word(self, word)
"""
    The method is used to count how many times a word exits in a file
    ...
    Attributes
    ----------
    word: str
        The read word used to update dictionary self.word2count
"""

def add_sentence(self, sentence)
"""
    The method is used to split sentence into tokens and upate the dictionary
    self.word2count
    ...
    Attributes
    ----------
    sentence: str
        The read sentence used to update dictionary self.word2count
"""

def set_word_embeddings(self)
"""
    The method is used to utilise the torch.nn.Embedding to generate word embeddings(embedding layer)
"""

def set_word_vector(self, word_vector)
"""
    The method is used to set the self.word_vector
    ...
    Attributes
    ----------
    word_vector: torch
        The word embedding vectors
"""

def from_word2vect_wordEmbeddings(self, freeze)
"""
    The method is used to create word embeddings(embedding layer) by the usage of word2vec dictionary and a string which indicates if the word embeddings(embedding layer) needs to freeze or not
    ...
    Attributes
    ----------
    freeze: bool
        The freeze parameter
"""

def from_word2vect_word2ind(self)
"""
    Convert the word2vec dictionary to word2ind by select all keys of word2vec 
    and index them in word2ind
"""

def set_word2vector(self)
"""
    Set the word2vec dictionary
"""

def filter(self, threshold)
"""
    Only select the words which appear to exceed the set value in the file and get their word embeddings(embedding layer)
    ...
    Attributes
    ----------
    threshold: int
        the lowerbound of the number of occurrences
"""

def get_word_embeddings(self)
"""
    Return the word_embeddings(embedding layer) which has been created
    ...
    Returns
    ----------
    self.word_embeddings: nn
        word_embeddings function
"""

def get_word_vector(self, word)
"""
    Return the word embedding according to input word
    ...
    Attributes
    ----------
    word: str
        the input word

    Returns
    ----------
    self.word2vec[word]: torch
        word embedding vectors of input word
"""

def get_word2vector(self)
"""
     Return self.word2vec dictionary
     ...
     Returns
    ----------
    self.word2vec: {str: torch}
        self.word2vec dictionary
"""

def get_sentence_ind(self, sentence, method)
"""
     Convert input sentence into word embeddings according to different models
     ...
    Attributes
    ----------
    sentence: str
        Input sentence
    method: str
        model name, Bow or Bilstm

    Returns
    ----------
    output: torch
        sentence word embeddings
"""

def setup(self, cor)
"""
    Convert the input original file into data that matches the settings
    ...
    Attributes
    ----------
    cor: str
        The file path of input original file
"""
```
