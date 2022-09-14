import sys
from collections import Counter
import numpy as np
import os
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
import torch.nn as nn
from sklearn.model_selection import PredefinedSplit

from torch_rnn_classifier import TorchRNNClassifier
from torch_tree_nn import TorchTreeNN
import sst
import utils

f = open('homework2_print.txt', 'w+')
sys.stdout = f
SST_HOME = os.path.join('data', 'sentiment')
sst_train = sst.train_reader(SST_HOME)
sst_dev = sst.dev_reader(SST_HOME)
bakeoff_dev = sst.bakeoff_dev_reader(SST_HOME)

from transformers import BertModel, BertTokenizer
import vsm

bert_weights_name = 'bert-base-uncased'



def unigrams_phi(text):
    return Counter(text.split())





def fit_softmax_classifier(X, y):
    mod = LogisticRegression(
        fit_intercept=True,
        solver='liblinear',
        multi_class='ovr')
    mod.fit(X, y)
    return mod


def rnn_phi(text):
    return text.split()


def fit_rnn_classifier(X, y):
    sst_glove_vocab = utils.get_vocab(X, mincount=2)
    mod = TorchRNNClassifier(
        sst_glove_vocab,
        early_stopping=True)
    mod.fit(X, y)
    return mod


# rnn_experiment = sst.experiment(
#     sst.train_reader(SST_HOME),
#     rnn_phi,
#     fit_rnn_classifier,
#     vectorize=False,  # For deep learning, use `vectorize=False`.
#     assess_dataframes=[sst_dev, bakeoff_dev])
#
#
# softmax_experiment = sst.experiment(
#     sst.train_reader(SST_HOME),   # Train on any data you like except SST-3 test!
#     unigrams_phi,                 # Free to write your own!
#     fit_softmax_classifier,       # Free to write your own!
#     assess_dataframes=[sst_dev, bakeoff_dev])  # Free to change this during development!



# def predict_one_softmax(text):
#     # Singleton list of feature dicts:
#     feats = [softmax_experiment['phi'](text)]
#     # Vectorize to get a feature matrix:
#     X = softmax_experiment['train_dataset']['vectorizer'].transform(feats)
#     # Standard sklearn `predict` step:
#     preds = softmax_experiment['model'].predict(X)
#     # Be sure to return the only member of the predictions,
#     # rather than the singleton list:
#     return preds[0]
#
#
# def predict_one_rnn(text):
#     # List of tokenized examples:
#     X = [rnn_experiment['phi'](text)]
#     # Standard `predict` step on a list of lists of str:
#     preds = rnn_experiment['model'].predict(X)
#     # Be sure to return the only member of the predictions,
#     # rather than the singleton list:
#     return preds[0]


def create_bakeoff_submission(
        predict_one_func,
        output_filename='cs224u-sentiment-bakeoff-entry.csv'):

    bakeoff_test = sst.bakeoff_test_reader(SST_HOME)
    sst_test = sst.test_reader(SST_HOME)
    bakeoff_test['dataset'] = 'bakeoff'
    sst_test['dataset'] = 'sst3'
    df = pd.concat((bakeoff_test, sst_test))

    df['prediction'] = df['sentence'].apply(predict_one_func)

    df.to_csv(output_filename, index=None)


def myownpredict():
    sst_train = sst.train_reader(SST_HOME)
    sst_dev = sst.dev_reader(SST_HOME)
    bakeoff_dev = sst.bakeoff_dev_reader(SST_HOME, include_subtrees=True)

    from torch_shallow_neural_classifier import TorchShallowNeuralClassifier

    bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)
    bert_model = BertModel.from_pretrained(bert_weights_name)

    def bert_phi(sen):
        ids = bert_tokenizer.batch_encode_plus(
            sen,
            add_special_tokens=True,
            return_attention_mask=True,
            padding='longest')
        X_example = torch.tensor(ids['input_ids'])
        X_example_mask = torch.tensor(ids['attention_mask'])

        with torch.no_grad():
            reps = bert_model(X_example, attention_mask=X_example_mask)
            return reps.last_hidden_state.squeeze(0).numpy()

    def bert_classifier_phi(sen):
        # reps = bert_phi(sen)
        # return reps.mean(axis=0)  # Another good, easy option.
        # return reps[0]
        # return sen
        toks = sen.lower().split()
        left = [utils.START_SYMBOL] + toks
        right = toks + [utils.END_SYMBOL]
        grams = list(zip(left, right))
        return Counter(grams)

    def fit_shallow_neural_classifier_with_hyperparameter_search(X, y):
        pass
        basemod = TorchShallowNeuralClassifier(
            early_stopping=True)
        cv = 5
        param_grid = {
            'hidden_dim': [100, 200, 300],
            'hidden_activation': [nn.Tanh(), nn.ReLU()]}
        bestmod = utils.fit_classifier_with_hyperparameter_search(
            X, y, basemod, cv, param_grid)
        return bestmod

    bert_classifier_xval = sst.experiment(
        sst_train,
        unigrams_phi,
        fit_shallow_neural_classifier_with_hyperparameter_search,
        assess_dataframes=sst_dev)

    optimized_bert_classifier = bert_classifier_xval['model']
    print('hyperparameter search finished')

    del bert_classifier_xval

    def fit_optimized_hf_bert_classifier(X, y):
        optimized_bert_classifier.max_iter = 1000
        optimized_bert_classifier.fit(X, y)
        return optimized_bert_classifier

    final_reps = sst.experiment(
        sst_train,
        bert_classifier_phi,
        fit_optimized_hf_bert_classifier,
        assess_dataframes=sst_dev,
        score_func=utils.safe_macro_f1,
        verbose=True)  # Pass in the BERT hidden state directly!

    return final_reps


softmax_experiment = myownpredict()


def predict_one_softmax(text):
    # Singleton list of feature dicts:
    feats = [softmax_experiment['phi'](text)]
    # Vectorize to get a feature matrix:
    X = softmax_experiment['train_dataset']['vectorizer'].transform(feats)
    # Standard sklearn `predict` step:
    preds = softmax_experiment['model'].predict(X)
    # Be sure to return the only member of the predictions,
    # rather than the singleton list:
    return preds[0]


if __name__ == '__main__':
    if 'IS_GRADESCOPE_ENV' not in os.environ:
        pass
        create_bakeoff_submission(predict_one_softmax)
        f.close()

