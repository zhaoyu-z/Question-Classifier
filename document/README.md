#### Tutorial about how to use the classifier
The classifier is used to do a task such as

```Input``` a question (e.g., "When was the first Wall Street Journal published ?")

``` Output``` one of N predefined classes (e.g., NUM:date, which is the class label for questions that require an answer of date)


#### How to setup environment 
(Ensure you have installed the basic python3 environment)


``` shell
# Install pytorch library if you do not
pip3 install --user pytorch

# Install sklearn library if you do not
pip3 install --user scikit-learn
```

#### Folders descriptions
```document```: a document containing a description for each function, a README file with instructions on how to use the code.

```data```: training, dev, test, configuration files and some extra files needed for models (e.g., vocabulary). For each model, one configuration file is provided(e.g., bow.config, bilstm.config).

```src```: source code.

#### How to use the classifier
Dowload all the three folders and move to ```src``` folder.

``` shell
cd src
```

Below are the **training steps** , every training needs few minutes to complete.

``` shell
# If you want to train the Bilstm model
python3 question_classifier.py --train --config '../data/bilstm.config'

# If you want to train the Bag of Words model
python3 question_classifier.py --train --config '../data/bow.config'
```

Below are the **testing steps**, **REMEMBER** you should run the corresponding training step of the classified model above firstly, and for the **Ensemble model**, you need to train both classified models firstly, i.e. training the Bilstm and Bow models at first.

``` shell
# If you want to test the Bilstm model
python3 question_classifier.py --test --config '../data/bilstm.config'

# If you want to test the Bag of Words model
python3 question_classifier.py --test --config '../data/bow.config'

# If you want to test the ensemble model
python3 question_classifier.py --test --config '../data/ensemble.config'
```

### How to adjust the parameters of both classifiers
Move to the ```data``` folder

``` shell
cd data

```

Open the corresponding configuration file for bilstm and bow model and the below parameters are allowed to change.

``` shell
## If pretrained is False, Randomly initialised word embeddings are set; 
## If pretrained is True, Pre-trained word embeddings are set.
pretrained : True

## If freeze is false, model fine-tune the pre-trained word embeddings.
## If freeze is True, model freeze the pre-trained word embeddings.
freeze : True

## The number of training epochs setting for the model
epoch: 10

## The learning rate used for training the models.
lr_param: 0.02

```




