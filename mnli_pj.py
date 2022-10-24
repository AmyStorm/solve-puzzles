import math
import random
import sys
import time

import torch
import numpy as np
import os
import pandas as pd
from datasets import load_dataset

from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer, get_constant_schedule_with_warmup
import vsm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

import nli
import utils

import logging

t = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
log = logging.getLogger()
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
file_handler = logging.FileHandler(f'nli_print-{t}.txt')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
log.addHandler(stdout_handler)
log.addHandler(file_handler)
# logging.basicConfig(filename=f'nli_print-{t}.txt',
#                     format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#                     level=logging.DEBUG)

from datasets import load_dataset


import torch.nn as nn

DATA_HOME = os.path.join("data", "nlidata")

ANNOTATIONS_HOME = os.path.join(DATA_HOME, "multinli_1.0_annotations")



GLOVE_HOME = os.path.join('data', 'glove.6B')

SEED = 1111
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Instantiate a Bert model and tokenizer based on `bert_weights_name`:
bert_weights_name = 'bert-base-uncased'
bert_model = BertModel.from_pretrained(bert_weights_name)
bert_tokenizer = BertTokenizer.from_pretrained(bert_weights_name)

example_texts = [
    "[CLS] {} [SEP] {} [SEP]"]

# snli_labels = ["contradiction", "entailment", "neutral"]


example_ids = bert_tokenizer.batch_encode_plus(
    example_texts,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='longest')

max_input_length = 256


def tokenize_bert(sentence):
    tokens = bert_tokenizer.tokenize(sentence)
    return tokens


def split_and_cut(sentence):
    tokens = sentence.strip().split(" ")
    tokens = tokens[:max_input_length]
    return tokens


def trim_sentence(sent):
    try:
        sent = sent.split()
        sent = sent[:128]
        return " ".join(sent)
    except:
        return sent


#Get list of 0s
def get_sent1_token_type(sent):
    try:
        return [0]* len(sent)
    except:
        return []


#Get list of 1s
def get_sent2_token_type(sent):
    try:
        return [1]* len(sent)
    except:
        return []


#combine from lists
def combine_seq(seq):
    return " ".join(seq)


#combines from lists of int
def combine_mask(mask):
    mask = [str(m) for m in mask]
    return " ".join(mask)


# def bert_phi(text):
#     input_ids = bert_tokenizer.encode(text, add_special_tokens=True)
#     X = torch.tensor([input_ids], device=device)
#     with torch.no_grad():
#         reps = bert_model(X)
#         return reps.last_hidden_state.squeeze(0).numpy()
#
#
# def bert_classifier_phi(text):
#     reps = bert_phi(text)
#     return reps.mean(axis=0)


# def temps():
#     input_ids = bert_tokenizer.encode(text, add_special_tokens=True)


# def word_cross_product_phi(ex):
#     """
#     Basis for cross-product features. Downcases all words.
#
#     Parameters
#     ----------
#     ex: NLIExample instance
#
#     Returns
#     -------
#     defaultdict
#         Maps each (w1, w2) in the cross-product of `t1.leaves()` and
#         `t2.leaves()` (both downcased) to its count. This is a
#         multi-set cross-product (repetitions matter).
#
#     """
#     words1 = [w.lower() for w in tokenizer.tokenize(ex.premise)]
#     words2 = [w.lower() for w in tokenizer.tokenize(ex.hypothesis)]
#     return Counter([(w1, w2) for w1, w2 in product(words1, words2)])
#
#
# def fit_softmax_with_hyperparameter_search(X, y):
#     """
#     A MaxEnt model of dataset with hyperparameter cross-validation.
#
#     Parameters
#     ----------
#     X : 2d np.array
#         The matrix of features, one example per row.
#
#     y : list
#         The list of labels for rows in `X`.
#
#     Returns
#     -------
#     sklearn.linear_model.LogisticRegression
#         A trained model instance, the best model found.
#
#     """
#
#     mod = LogisticRegression(
#         fit_intercept=True,
#         max_iter=3,  ## A small number of iterations.
#         solver='liblinear',
#         multi_class='ovr')
#
#     param_grid = {
#         'C': [0.4, 0.6, 0.8, 1.0],
#         'penalty': ['l1','l2']}
#
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         bestmod = utils.fit_classifier_with_hyperparameter_search(
#             X, y, mod, param_grid=param_grid, cv=3)
#
#     return bestmod


def snli_csv_gen():
    snli = load_dataset("snli")
    log.debug(torch.cuda.is_available())
    log.debug(snli.keys())

    # snli_trains = pd.DataFrame(columns=['premise', 'hypothesis', 'label', 'sequence', 'attention_mask', 'token_type'])
    # nli.NLIReader(snli['train'], samp_percentage=0.10, random_state=42)
    snli_trains_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(snli['train'], filter_unlabeled=True).read()])
    snli_trains_hypothesis = pd.Series([ex.hypothesis
                                           for ex in nli.NLIReader(snli['train'], filter_unlabeled=True).read()])
    snli_trains_label = pd.Series([ex.label
                                      for ex in nli.NLIReader(snli['train'], filter_unlabeled=True).read()])
    snli_valid_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(snli['validation'], filter_unlabeled=True).read()])
    snli_valid_hypothesis = pd.Series([ex.hypothesis
                                        for ex in nli.NLIReader(snli['validation'], filter_unlabeled=True).read()])
    snli_valid_label = pd.Series([ex.label
                                        for ex in nli.NLIReader(snli['validation'], filter_unlabeled=True).read()])
    snli_test_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(snli['test'], filter_unlabeled=True).read()])
    snli_test_hypothesis = pd.Series([ex.hypothesis
                                        for ex in nli.NLIReader(snli['test'], filter_unlabeled=True).read()])
    snli_test_label = pd.Series([ex.label
                                        for ex in nli.NLIReader(snli['test'], filter_unlabeled=True).read()])

    snli_trains = pd.DataFrame()
    snli_trains.insert(loc=0, column='premise', value=snli_trains_premise)
    snli_trains.insert(loc=1, column='hypothesis', value=snli_trains_hypothesis)
    snli_trains.insert(loc=2, column='label', value=snli_trains_label)
    snli_valid = pd.DataFrame()
    snli_valid.insert(loc=0, column='premise', value=snli_valid_premise)
    snli_valid.insert(loc=1, column='hypothesis', value=snli_valid_hypothesis)
    snli_valid.insert(loc=2, column='label', value=snli_valid_label)
    snli_test = pd.DataFrame()
    snli_test.insert(loc=0, column='premise', value=snli_test_premise)
    snli_test.insert(loc=1, column='hypothesis', value=snli_test_hypothesis)
    snli_test.insert(loc=2, column='label', value=snli_test_label)
    # snli_trains = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['train'], filter_unlabeled=True).read()])
    #
    # snli_valid = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['validation'], filter_unlabeled=True).read()])
    #
    # snli_test = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['test'], filter_unlabeled=True).read()])

    ## decrease datas
    # snli_trains = snli_trains[:10]
    # snli_valid = snli_valid[:1]
    # snli_test = snli_test[:1]


    # Add [CLS] and [SEP] tokens
    seq_train_1 = np.full((snli_trains[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                   snli_trains[['premise']].apply(trim_sentence) + \
                   np.full((snli_trains[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_train_2 = snli_trains[['hypothesis']].apply(trim_sentence) + \
                   np.full((snli_trains[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)
    seq_dev_1 = np.full((snli_valid[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                 snli_valid[['premise']].apply(trim_sentence) + \
                 np.full((snli_valid[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_dev_2 = snli_valid[['hypothesis']].apply(trim_sentence) + \
                 np.full((snli_valid[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)
    seq_test_1 = np.full((snli_test[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                  snli_test[['premise']].apply(trim_sentence) + \
                  np.full((snli_test[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_test_2 = snli_test[['hypothesis']].apply(trim_sentence) + \
                  np.full((snli_test[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)


    # Apply Bert Tokenizer for tokeinizing
    tk_train_1 = seq_train_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_train_2 = seq_train_2.T.apply(lambda x: tokenize_bert(x[0]))
    tk_dev_1 = seq_dev_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_dev_2 = seq_dev_2.T.apply(lambda x: tokenize_bert(x[0]))
    tk_test_1 = seq_test_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_test_2 = seq_test_2.T.apply(lambda x: tokenize_bert(x[0]))


    tkt_train_1 = tk_train_1.apply(lambda x: get_sent1_token_type(x))
    tkt_train_2 = tk_train_2.apply(lambda x: get_sent2_token_type(x))
    tkt_dev_1 = tk_dev_1.apply(lambda x: get_sent1_token_type(x))
    tkt_dev_2 = tk_dev_2.apply(lambda x: get_sent2_token_type(x))
    tkt_test_1 = tk_test_1.apply(lambda x: get_sent1_token_type(x))
    tkt_test_2 = tk_test_2.apply(lambda x: get_sent2_token_type(x))

    # Combine both sequences
    snli_trains_seq = tk_train_1 + tk_train_2
    snli_valid_seq = tk_dev_1 + tk_dev_2
    snli_test_seq = tk_test_1 + tk_test_2

    # Get attention mask
    snli_trains_at = snli_trains_seq.apply(lambda x: get_sent2_token_type(x))
    snli_valid_at = snli_valid_seq.apply(lambda x: get_sent2_token_type(x))
    snli_test_at = snli_test_seq.apply(lambda x: get_sent2_token_type(x))

    # Get combined token type ids for input
    snli_trains_tt = tkt_train_1 + tkt_train_2
    snli_valid_tt = tkt_dev_1 + tkt_dev_2
    snli_test_tt = tkt_test_1 + tkt_test_2

    # Now make all these inputs as sequential data to be easily fed into torchtext Field.
    snli_trains.insert(loc=3, column='sequence', value=snli_trains_seq.apply(combine_seq))
    snli_valid.insert(loc=3, column='sequence', value=snli_valid_seq.apply(combine_seq))
    snli_test.insert(loc=3, column='sequence', value=snli_test_seq.apply(combine_seq))
    snli_trains.insert(loc=4, column='attention_mask', value=snli_trains_at.apply(combine_mask))
    snli_valid.insert(loc=4, column='attention_mask', value=snli_valid_at.apply(combine_mask))
    snli_test.insert(loc=4, column='attention_mask', value=snli_test_at.apply(combine_mask))
    snli_trains.insert(loc=5, column='token_type', value=snli_trains_tt.apply(combine_mask))
    snli_valid.insert(loc=5, column='token_type', value=snli_valid_tt.apply(combine_mask))
    snli_test.insert(loc=5, column='token_type', value=snli_test_tt.apply(combine_mask))

    # snli_trains = snli_trains.drop(columns='premise')
    # snli_trains = snli_trains.drop(columns='hypothesis')
    # snli_valid = snli_valid.drop(columns='premise')
    # snli_valid = snli_valid.drop(columns='hypothesis')
    # snli_test = snli_test.drop(columns='premise')
    # snli_test = snli_test.drop(columns='hypothesis')
    # snli_iterator = iter(nli.NLIReader(snli['train']).read())
    # snli_ex = next(snli_iterator)
    # print(snli_ex)
    # snli_ex["hypothesis"]
    # snli_ex["premise"]
    # snli_ex["label"]

    log.debug(snli_trains)
    os.makedirs('snli_1.0', exist_ok=True)
    snli_trains.to_csv('snli_1.0/snli_1.0_train.csv', index=False)
    snli_valid.to_csv('snli_1.0/snli_1.0_dev.csv', index=False)
    snli_test.to_csv('snli_1.0/snli_1.0_test.csv', index=False)


def mnli_csv_gen():
    mnli = load_dataset("multi_nli")
    log.info(mnli.keys())

    # snli_trains = pd.DataFrame(columns=['premise', 'hypothesis', 'label', 'sequence', 'attention_mask', 'token_type'])
    # nli.NLIReader(snli['train'], samp_percentage=0.10, random_state=42)
    snli_trains_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(mnli['train'], filter_unlabeled=True).read()])
    snli_trains_hypothesis = pd.Series([ex.hypothesis
                                           for ex in nli.NLIReader(mnli['train'], filter_unlabeled=True).read()])
    snli_trains_label = pd.Series([ex.label
                                      for ex in nli.NLIReader(mnli['train'], filter_unlabeled=True).read()])
    snli_valid_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(mnli['validation_matched'], filter_unlabeled=True).read()])
    snli_valid_hypothesis = pd.Series([ex.hypothesis
                                        for ex in nli.NLIReader(mnli['validation_matched'], filter_unlabeled=True).read()])
    snli_valid_label = pd.Series([ex.label
                                        for ex in nli.NLIReader(mnli['validation_matched'], filter_unlabeled=True).read()])
    snli_test_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(mnli['validation_mismatched'], filter_unlabeled=True).read()])
    snli_test_hypothesis = pd.Series([ex.hypothesis
                                        for ex in nli.NLIReader(mnli['validation_mismatched'], filter_unlabeled=True).read()])
    snli_test_label = pd.Series([ex.label
                                        for ex in nli.NLIReader(mnli['validation_mismatched'], filter_unlabeled=True).read()])

    snli_trains = pd.DataFrame()
    snli_trains.insert(loc=0, column='premise', value=snli_trains_premise)
    snli_trains.insert(loc=1, column='hypothesis', value=snli_trains_hypothesis)
    snli_trains.insert(loc=2, column='label', value=snli_trains_label)
    snli_valid = pd.DataFrame()
    snli_valid.insert(loc=0, column='premise', value=snli_valid_premise)
    snli_valid.insert(loc=1, column='hypothesis', value=snli_valid_hypothesis)
    snli_valid.insert(loc=2, column='label', value=snli_valid_label)
    snli_test = pd.DataFrame()
    snli_test.insert(loc=0, column='premise', value=snli_test_premise)
    snli_test.insert(loc=1, column='hypothesis', value=snli_test_hypothesis)
    snli_test.insert(loc=2, column='label', value=snli_test_label)
    # snli_trains = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['train'], filter_unlabeled=True).read()])
    #
    # snli_valid = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['validation'], filter_unlabeled=True).read()])
    #
    # snli_test = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['test'], filter_unlabeled=True).read()])

    ## decrease datas
    # snli_trains = snli_trains[:10]
    # snli_valid = snli_valid[:1]
    # snli_test = snli_test[:1]


    # Add [CLS] and [SEP] tokens
    seq_train_1 = np.full((snli_trains[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                   snli_trains[['premise']].apply(trim_sentence) + \
                   np.full((snli_trains[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_train_2 = snli_trains[['hypothesis']].apply(trim_sentence) + \
                   np.full((snli_trains[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)
    seq_dev_1 = np.full((snli_valid[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                 snli_valid[['premise']].apply(trim_sentence) + \
                 np.full((snli_valid[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_dev_2 = snli_valid[['hypothesis']].apply(trim_sentence) + \
                 np.full((snli_valid[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)
    seq_test_1 = np.full((snli_test[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                  snli_test[['premise']].apply(trim_sentence) + \
                  np.full((snli_test[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_test_2 = snli_test[['hypothesis']].apply(trim_sentence) + \
                  np.full((snli_test[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)


    # Apply Bert Tokenizer for tokeinizing
    tk_train_1 = seq_train_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_train_2 = seq_train_2.T.apply(lambda x: tokenize_bert(x[0]))
    tk_dev_1 = seq_dev_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_dev_2 = seq_dev_2.T.apply(lambda x: tokenize_bert(x[0]))
    tk_test_1 = seq_test_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_test_2 = seq_test_2.T.apply(lambda x: tokenize_bert(x[0]))


    tkt_train_1 = tk_train_1.apply(lambda x: get_sent1_token_type(x))
    tkt_train_2 = tk_train_2.apply(lambda x: get_sent2_token_type(x))
    tkt_dev_1 = tk_dev_1.apply(lambda x: get_sent1_token_type(x))
    tkt_dev_2 = tk_dev_2.apply(lambda x: get_sent2_token_type(x))
    tkt_test_1 = tk_test_1.apply(lambda x: get_sent1_token_type(x))
    tkt_test_2 = tk_test_2.apply(lambda x: get_sent2_token_type(x))

    # Combine both sequences
    snli_trains_seq = tk_train_1 + tk_train_2
    snli_valid_seq = tk_dev_1 + tk_dev_2
    snli_test_seq = tk_test_1 + tk_test_2

    # Get attention mask
    snli_trains_at = snli_trains_seq.apply(lambda x: get_sent2_token_type(x))
    snli_valid_at = snli_valid_seq.apply(lambda x: get_sent2_token_type(x))
    snli_test_at = snli_test_seq.apply(lambda x: get_sent2_token_type(x))

    # Get combined token type ids for input
    snli_trains_tt = tkt_train_1 + tkt_train_2
    snli_valid_tt = tkt_dev_1 + tkt_dev_2
    snli_test_tt = tkt_test_1 + tkt_test_2

    # Now make all these inputs as sequential data to be easily fed into torchtext Field.
    snli_trains.insert(loc=3, column='sequence', value=snli_trains_seq.apply(combine_seq))
    snli_valid.insert(loc=3, column='sequence', value=snli_valid_seq.apply(combine_seq))
    snli_test.insert(loc=3, column='sequence', value=snli_test_seq.apply(combine_seq))
    snli_trains.insert(loc=4, column='attention_mask', value=snli_trains_at.apply(combine_mask))
    snli_valid.insert(loc=4, column='attention_mask', value=snli_valid_at.apply(combine_mask))
    snli_test.insert(loc=4, column='attention_mask', value=snli_test_at.apply(combine_mask))
    snli_trains.insert(loc=5, column='token_type', value=snli_trains_tt.apply(combine_mask))
    snli_valid.insert(loc=5, column='token_type', value=snli_valid_tt.apply(combine_mask))
    snli_test.insert(loc=5, column='token_type', value=snli_test_tt.apply(combine_mask))

    # snli_trains = snli_trains.drop(columns='premise')
    # snli_trains = snli_trains.drop(columns='hypothesis')
    # snli_valid = snli_valid.drop(columns='premise')
    # snli_valid = snli_valid.drop(columns='hypothesis')
    # snli_test = snli_test.drop(columns='premise')
    # snli_test = snli_test.drop(columns='hypothesis')
    # snli_iterator = iter(nli.NLIReader(snli['train']).read())
    # snli_ex = next(snli_iterator)
    # print(snli_ex)
    # snli_ex["hypothesis"]
    # snli_ex["premise"]
    # snli_ex["label"]

    log.debug(snli_trains)
    os.makedirs('mnli_1.0', exist_ok=True)
    snli_trains.to_csv('mnli_1.0/mnli_train.csv', index=False)
    snli_valid.to_csv('mnli_1.0/mnli_validation_matched.csv', index=False)
    snli_test.to_csv('mnli_1.0/mnli_validation_mismatched.csv', index=False)


def impli_csv_gen():
    mnli = load_dataset("multi_nli")
    log.info(mnli.keys())


    snli_trains_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(mnli['train'], filter_unlabeled=True).read()])
    snli_trains_hypothesis = pd.Series([ex.hypothesis
                                           for ex in nli.NLIReader(mnli['train'], filter_unlabeled=True).read()])
    snli_trains_label = pd.Series([ex.label
                                      for ex in nli.NLIReader(mnli['train'], filter_unlabeled=True).read()])
    snli_valid_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(mnli['validation_matched'], filter_unlabeled=True).read()])
    snli_valid_hypothesis = pd.Series([ex.hypothesis
                                        for ex in nli.NLIReader(mnli['validation_matched'], filter_unlabeled=True).read()])
    snli_valid_label = pd.Series([ex.label
                                        for ex in nli.NLIReader(mnli['validation_matched'], filter_unlabeled=True).read()])
    snli_test_premise = pd.Series([ex.premise
                                        for ex in nli.NLIReader(mnli['validation_mismatched'], filter_unlabeled=True).read()])
    snli_test_hypothesis = pd.Series([ex.hypothesis
                                        for ex in nli.NLIReader(mnli['validation_mismatched'], filter_unlabeled=True).read()])
    snli_test_label = pd.Series([ex.label
                                        for ex in nli.NLIReader(mnli['validation_mismatched'], filter_unlabeled=True).read()])

    snli_trains = pd.DataFrame()
    snli_trains.insert(loc=0, column='premise', value=snli_trains_premise)
    snli_trains.insert(loc=1, column='hypothesis', value=snli_trains_hypothesis)
    snli_trains.insert(loc=2, column='label', value=snli_trains_label)
    snli_valid = pd.DataFrame()
    snli_valid.insert(loc=0, column='premise', value=snli_valid_premise)
    snli_valid.insert(loc=1, column='hypothesis', value=snli_valid_hypothesis)
    snli_valid.insert(loc=2, column='label', value=snli_valid_label)
    snli_test = pd.DataFrame()
    snli_test.insert(loc=0, column='premise', value=snli_test_premise)
    snli_test.insert(loc=1, column='hypothesis', value=snli_test_hypothesis)
    snli_test.insert(loc=2, column='label', value=snli_test_label)
    # snli_trains = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['train'], filter_unlabeled=True).read()])
    #
    # snli_valid = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['validation'], filter_unlabeled=True).read()])
    #
    # snli_test = pd.Series(
    #     [ex.label for ex in nli.NLIReader(
    #         snli['test'], filter_unlabeled=True).read()])

    ## decrease datas
    # snli_trains = snli_trains[:10]
    # snli_valid = snli_valid[:1]
    # snli_test = snli_test[:1]


    # Add [CLS] and [SEP] tokens
    seq_train_1 = np.full((snli_trains[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                   snli_trains[['premise']].apply(trim_sentence) + \
                   np.full((snli_trains[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_train_2 = snli_trains[['hypothesis']].apply(trim_sentence) + \
                   np.full((snli_trains[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)
    seq_dev_1 = np.full((snli_valid[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                 snli_valid[['premise']].apply(trim_sentence) + \
                 np.full((snli_valid[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_dev_2 = snli_valid[['hypothesis']].apply(trim_sentence) + \
                 np.full((snli_valid[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)
    seq_test_1 = np.full((snli_test[['premise']].size, 1), bert_tokenizer.cls_token + ' ') + \
                  snli_test[['premise']].apply(trim_sentence) + \
                  np.full((snli_test[['premise']].size, 1), ' ' + bert_tokenizer.sep_token + ' ')
    seq_test_2 = snli_test[['hypothesis']].apply(trim_sentence) + \
                  np.full((snli_test[['hypothesis']].size, 1), ' ' + bert_tokenizer.sep_token)


    # Apply Bert Tokenizer for tokeinizing
    tk_train_1 = seq_train_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_train_2 = seq_train_2.T.apply(lambda x: tokenize_bert(x[0]))
    tk_dev_1 = seq_dev_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_dev_2 = seq_dev_2.T.apply(lambda x: tokenize_bert(x[0]))
    tk_test_1 = seq_test_1.T.apply(lambda x: tokenize_bert(x[0]))
    tk_test_2 = seq_test_2.T.apply(lambda x: tokenize_bert(x[0]))


    tkt_train_1 = tk_train_1.apply(lambda x: get_sent1_token_type(x))
    tkt_train_2 = tk_train_2.apply(lambda x: get_sent2_token_type(x))
    tkt_dev_1 = tk_dev_1.apply(lambda x: get_sent1_token_type(x))
    tkt_dev_2 = tk_dev_2.apply(lambda x: get_sent2_token_type(x))
    tkt_test_1 = tk_test_1.apply(lambda x: get_sent1_token_type(x))
    tkt_test_2 = tk_test_2.apply(lambda x: get_sent2_token_type(x))

    # Combine both sequences
    snli_trains_seq = tk_train_1 + tk_train_2
    snli_valid_seq = tk_dev_1 + tk_dev_2
    snli_test_seq = tk_test_1 + tk_test_2

    # Get attention mask
    snli_trains_at = snli_trains_seq.apply(lambda x: get_sent2_token_type(x))
    snli_valid_at = snli_valid_seq.apply(lambda x: get_sent2_token_type(x))
    snli_test_at = snli_test_seq.apply(lambda x: get_sent2_token_type(x))

    # Get combined token type ids for input
    snli_trains_tt = tkt_train_1 + tkt_train_2
    snli_valid_tt = tkt_dev_1 + tkt_dev_2
    snli_test_tt = tkt_test_1 + tkt_test_2

    # Now make all these inputs as sequential data to be easily fed into torchtext Field.
    snli_trains.insert(loc=3, column='sequence', value=snli_trains_seq.apply(combine_seq))
    snli_valid.insert(loc=3, column='sequence', value=snli_valid_seq.apply(combine_seq))
    snli_test.insert(loc=3, column='sequence', value=snli_test_seq.apply(combine_seq))
    snli_trains.insert(loc=4, column='attention_mask', value=snli_trains_at.apply(combine_mask))
    snli_valid.insert(loc=4, column='attention_mask', value=snli_valid_at.apply(combine_mask))
    snli_test.insert(loc=4, column='attention_mask', value=snli_test_at.apply(combine_mask))
    snli_trains.insert(loc=5, column='token_type', value=snli_trains_tt.apply(combine_mask))
    snli_valid.insert(loc=5, column='token_type', value=snli_valid_tt.apply(combine_mask))
    snli_test.insert(loc=5, column='token_type', value=snli_test_tt.apply(combine_mask))

    # snli_trains = snli_trains.drop(columns='premise')
    # snli_trains = snli_trains.drop(columns='hypothesis')
    # snli_valid = snli_valid.drop(columns='premise')
    # snli_valid = snli_valid.drop(columns='hypothesis')
    # snli_test = snli_test.drop(columns='premise')
    # snli_test = snli_test.drop(columns='hypothesis')
    # snli_iterator = iter(nli.NLIReader(snli['train']).read())
    # snli_ex = next(snli_iterator)
    # print(snli_ex)
    # snli_ex["hypothesis"]
    # snli_ex["premise"]
    # snli_ex["label"]

    log.debug(snli_trains)
    os.makedirs('mnli_1.0', exist_ok=True)
    snli_trains.to_csv('mnli_1.0/mnli_train.csv', index=False)
    snli_valid.to_csv('mnli_1.0/mnli_validation_matched.csv', index=False)
    snli_test.to_csv('mnli_1.0/mnli_validation_mismatched.csv', index=False)

# To convert back attention mask and token type ids to integer.
def convert_to_int(tok_ids):
    tok_ids = [int(x) for x in tok_ids]
    return tok_ids

def categorical_accuracy(preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)
        correct = (max_preds.squeeze(1) == y).float()
        return correct.sum() / len(y)

def train(model, iterator, optimizer, criterion, scheduler, epoch):
    # print(iterator)

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i, (labels, sequences, attention_masks, token_types) in enumerate(iterator):
        optimizer.zero_grad()  # clear gradients first
        torch.cuda.empty_cache()  # releases all unoccupied cached memory


        # print(sequence.size(), attn_mask.size(), token_type.size())
        # print(sequence[0])
        # print(attn_mask[0])
        # print(token_type[0])

        predictions = model(sequences, attention_masks, token_types)

        # predictions = [batch_size, 3]
        # print(predictions.size())

        loss = criterion(predictions, labels)

        acc = categorical_accuracy(predictions, labels)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        now = time.strftime('"%Y-%m-%d %H:%M:%S"', time.localtime())
        log.debug(f'{now} epoch {epoch}  batch {i} * BATCH_SIZE = {i * BATCH_SIZE}')

    return epoch_loss / len(iterator.dataset), epoch_acc / len(iterator.dataset)

def evaluate(model, iterator, criterion):
    # print(iterator)
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for i, (labels, sequences, attention_masks, token_types) in enumerate(iterator):
            # print(batch)

            predictions = model(sequences, attention_masks, token_types)

            loss = criterion(predictions, labels)

            acc = categorical_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator.dataset), epoch_acc / len(iterator.dataset)

def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

def get_scheduler(optimizer, warmup_steps):
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        return scheduler

fp16 = False
BATCH_SIZE = 32
max_grad_norm = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch.utils.data import DataLoader, random_split
from collections import Counter
from torchtext.vocab import vocab

def builditerator(df):
    # 単語分割
    df['sequence'] = df['sequence'].map(lambda x: bert_tokenizer.convert_tokens_to_ids(split_and_cut(x)))
    df['attention_mask'] = df['attention_mask'].map(lambda x: convert_to_int(split_and_cut(x)))
    df['token_type'] = df['token_type'].map(lambda x: convert_to_int(split_and_cut(x)))

    # ラベル辞書
    df['label'] = df['label'].astype(str)  # ラベルが数値のため文字型に変換。文字型で指定していた場合はこの処理は不要
    counter = Counter(df['label'])
    label_vocab = vocab(counter, specials=(['<unk>']))
    label_vocab.set_default_index(label_vocab['<unk>'])
    # DataLoader設定
    import torchtext.transforms as T

    # transform生成
    # text_transform = T.Sequential(T.ToTensor())
    label_transform = lambda x: label_vocab[x]

    # attention_mask_transform = T.Sequential(T.ToTensor())
    # token_type_transform = T.Sequential(T.ToTensor())

    def collate_batch(batch):
        sequences, labels, attention_masks, token_types = [], [], [], []
        for (label, sequence, attention_mask, token_type) in batch:
            sequences.append(sequence)
            labels.append(label_transform(label))
            attention_masks.append(attention_mask)
            token_types.append(token_type)
        return torch.tensor(labels).to(device), \
               pad_sequence([torch.tensor(s) for s in sequences], batch_first=True).to(device), \
               pad_sequence([torch.tensor(a) for a in attention_masks], batch_first=True).to(device), \
               pad_sequence([torch.tensor(t) for t in token_types], batch_first=True).to(device)

    def batch_sampler():
        indices = [(i, len(s[1])) for i, s in
                   enumerate(df[['label', 'sequence', 'attention_mask', 'token_type']].values.tolist())]
        random.shuffle(indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(indices), BATCH_SIZE * 100):
            pooled_indices.extend(sorted(indices[i:i + BATCH_SIZE * 100], key=lambda x: x[1]))

        pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        for i in range(0, len(pooled_indices), BATCH_SIZE):
            yield pooled_indices[i:i + BATCH_SIZE]

    data_loader = DataLoader(df[['label', 'sequence', 'attention_mask', 'token_type']].values.tolist(),
                             batch_sampler=batch_sampler(), collate_fn=collate_batch)
    return label_vocab, data_loader


class BERTNLIModel(nn.Module):
    def __init__(self,
                 bert_model,
                 hidden_dim,
                 output_dim, ):
        super().__init__()
        self.bert = bert_model

        embedding_dim = bert_model.config.to_dict()['hidden_size']
        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, sequence, attn_mask, token_type):
        embedded = self.bert(input_ids=sequence, attention_mask=attn_mask, token_type_ids=token_type)[1]
        output = self.out(embedded)
        return output

def snli_bert_exe():

    log.info('snli_bert_exe start')
    text_vocab, train_it = builditerator(pd.read_csv('snli_1.0/snli_1.0_train.csv'))
    text_vocab_dev, valid_it = builditerator(pd.read_csv('snli_1.0/snli_1.0_dev.csv'))
    text_vocab_test, test_it = builditerator(pd.read_csv('snli_1.0/snli_1.0_test.csv'))

    log.info(f"Number of train data: {len(train_it.dataset)}")
    log.info(f"Number of dev data: {len(valid_it.dataset)}")
    log.info(f"Number of test data: {len(test_it.dataset)}")

    # defining model
    HIDDEN_DIM = 512
    OUTPUT_DIM = len(text_vocab.vocab)
    model = BERTNLIModel(bert_model,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-6)

    criterion = nn.CrossEntropyLoss().to(device)

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    N_EPOCHS = 12
    train_data_len = len(train_it.dataset)
    warmup_percent = 0.2
    total_steps = math.ceil(N_EPOCHS * train_data_len * 1. / BATCH_SIZE)
    warmup_steps = int(total_steps * warmup_percent)
    scheduler = get_scheduler(optimizer, warmup_steps)

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_it, optimizer, criterion, scheduler, epoch)
        valid_loss, valid_acc = evaluate(model, valid_it, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'bert-nli.pt')

        log.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        log.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        log.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('bert-nli.pt'))

    test_loss, test_acc = evaluate(model, test_it, criterion)

    log.info(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')

    def predict_inference(premise, hypothesis, model, device):

        model.eval()

        premise = '[CLS] ' + premise + ' [SEP]'
        hypothesis = hypothesis + ' [SEP]'

        prem_t = tokenize_bert(premise)
        hypo_t = tokenize_bert(hypothesis)

        # print(len(prem_t), len(hypo_t))

        prem_type = get_sent1_token_type(prem_t)
        hypo_type = get_sent2_token_type(hypo_t)

        # print(len(prem_type), len(hypo_type))

        indexes = prem_t + hypo_t

        indexes = bert_tokenizer.convert_tokens_to_ids(indexes)
        # print(indexes)
        indexes_type = prem_type + hypo_type
        # print(indexes_type)

        attn_mask = get_sent2_token_type(indexes)
        # print(attn_mask)

        # print(len(indexes))
        # print(len(indexes_type))
        # print(len(attn_mask))

        # seq = '[CLS] '+ premise + ' [SEP] '+ hypothesis

        # tokens = tokenizer.tokenize(seq)

        # indexes = tokenizer.convert_tokens_to_ids(tokens)

        indexes = torch.LongTensor(indexes).unsqueeze(0).to(device)
        indexes_type = torch.LongTensor(indexes_type).unsqueeze(0).to(device)
        attn_mask = torch.LongTensor(attn_mask).unsqueeze(0).to(device)

        # print(indexes.size())

        prediction = model(indexes, attn_mask, indexes_type)

        prediction = prediction.argmax(dim=-1).item()

        return text_vocab.vocab.itos_[prediction]

    premise = 'a man sitting on a green bench.'
    hypothesis = 'a woman sitting on a green bench.'

    result = predict_inference(premise, hypothesis, model, device)
    log.info(f'premise:{premise}')
    log.info(f'hypothesis:{hypothesis}')
    log.info(f'result:{result}')
    premise = 'a man sitting on a green bench.'
    hypothesis = 'a man sitting on a blue bench.'

    predict_inference(premise, hypothesis, model, device)
    log.info(f'premise:{premise}')
    log.info(f'hypothesis:{hypothesis}')
    log.info(f'result:{result}')


def mnli_bert_exe():
    log.info('mnli_bert_exe start')
    # text_vocab, train_it = builditerator('mnli_1.0/mnli_train.csv')
    # text_vocab_dev, matched_it = builditerator('mnli_1.0/mnli_validation_matched.csv')
    # text_vocab_test, mismatched_it = builditerator('mnli_1.0/mnli_validation_mismatched.csv')

    text_vocab, train_it = builditerator(pd.read_csv('mnli_1.0/mnli_train.csv'))
    matched_it = pd.read_csv('mnli_1.0/mnli_validation_matched.csv')
    mismatched_it = pd.read_csv('mnli_1.0/mnli_validation_mismatched.csv')

    matched_len = matched_it.shape[0]
    mismatched_len = mismatched_it.shape[0]
    matched_valid_it, matched_test_it = random_split(
        dataset=matched_it[['label', 'sequence', 'attention_mask', 'token_type']].values.tolist(),
        lengths=[int(matched_len * 0.5), matched_len - int(matched_len * 0.5)],
        generator=torch.Generator().manual_seed(1)
    )

    mismatched_valid_it, mismatched_test_it = random_split(
        dataset=mismatched_it[['label', 'sequence', 'attention_mask', 'token_type']].values.tolist(),
        lengths=[int(mismatched_len * 0.5), mismatched_len - int(mismatched_len * 0.5)],
        generator=torch.Generator().manual_seed(1)
    )


    text_vocab_dev1, matched_valid_it = builditerator(
        pd.DataFrame(list(matched_valid_it), columns=['label', 'sequence', 'attention_mask', 'token_type']))
    text_vocab_dev2, matched_test_it = builditerator(
        pd.DataFrame(list(matched_test_it), columns=['label', 'sequence', 'attention_mask', 'token_type']))
    text_vocab_test1, mismatched_valid_it = builditerator(
        pd.DataFrame(list(mismatched_valid_it), columns=['label', 'sequence', 'attention_mask', 'token_type']))
    text_vocab_test2, mismatched_test_it = builditerator(
        pd.DataFrame(list(mismatched_test_it), columns=['label', 'sequence', 'attention_mask', 'token_type']))

    log.info(f"Number of train data: {len(train_it.dataset)}")
    log.info(f"Number of dev data: {len(matched_valid_it.dataset)}")
    log.info(f"Number of dev data: {len(matched_test_it.dataset)}")
    log.info(f"Number of test data: {len(mismatched_valid_it.dataset)}")
    log.info(f"Number of test data: {len(mismatched_test_it.dataset)}")

    # defining model
    HIDDEN_DIM = 512
    OUTPUT_DIM = len(text_vocab.vocab)
    model = BERTNLIModel(bert_model,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-6)

    criterion = nn.CrossEntropyLoss().to(device)

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    N_EPOCHS = 12
    train_data_len = len(train_it.dataset)
    warmup_percent = 0.2
    total_steps = math.ceil(N_EPOCHS * train_data_len * 1. / BATCH_SIZE)
    warmup_steps = int(total_steps * warmup_percent)
    scheduler = get_scheduler(optimizer, warmup_steps)

    best_valid_loss1 = float('inf')
    best_valid_loss2 = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss, train_acc = train(model, train_it, optimizer, criterion, scheduler, epoch)
        valid_loss, valid_acc = evaluate(model, matched_valid_it, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss1:
            best_valid_loss1 = valid_loss
            torch.save(model.state_dict(), 'bert-mnli-matched.pt')

        log.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        log.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        log.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        log.info(f'\t BEST Val. Loss: {best_valid_loss1:.3f}')

        mis_valid_loss, mis_valid_acc = evaluate(model, mismatched_valid_it, criterion)

        start_time = time.time()
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if mis_valid_loss < best_valid_loss2:
            best_valid_loss2 = mis_valid_loss
            torch.save(model.state_dict(), 'bert-mnli-mismatched.pt')

        log.info(f'MIS Epoch: {epoch + 1:02} | MIS Epoch Time: {epoch_mins}m {epoch_secs}s')
        log.info(f'\tMIS Train Loss: {train_loss:.3f} | MIS Train Acc: {train_acc * 100:.2f}%')
        log.info(f'\tMIS Val. Loss: {mis_valid_loss:.3f} |  MIS Val. Acc: {mis_valid_acc * 100:.2f}%')
        log.info(f'\t BEST MIS Val. Loss: {best_valid_loss2:.3f}')

    model.load_state_dict(torch.load('bert-mnli-matched.pt'))

    test_loss, test_acc = evaluate(model, matched_test_it, criterion)

    log.info(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')

    model.load_state_dict(torch.load('bert-mnli-mismatched.pt'))
    mis_test_loss, mis_test_acc = evaluate(model, mismatched_test_it, criterion)

    log.info(f'MIS Test Loss: {mis_test_loss:.3f} |  MIS Test Acc: {mis_test_acc * 100:.2f}%')


if __name__ == '__main__':
    # snli_csv_gen()
    # mnli_csv_gen()
    log.info('start')
    mnli_bert_exe()
    snli_bert_exe()

    # word_cross_product_experiment_xval = nli.experiment(
    #     train_reader=nli.NLIReader(snli['train']),
    #     phi=word_cross_product_phi,
    #     train_func=fit_softmax_with_hyperparameter_search,
    #     assess_reader=None,
    #     verbose=False)
    #
    # optimized_word_cross_product_model = word_cross_product_experiment_xval['model']
    #
    #
    # def fit_optimized_word_cross_product(X, y):
    #     optimized_word_cross_product_model.max_iter = 1000  # To convergence in this phase!
    #     optimized_word_cross_product_model.fit(X, y)
    #     return optimized_word_cross_product_model
    #
    #
    # _ = nli.experiment(
    #     train_reader=nli.NLIReader(snli['train']),
    #     phi=bert_classifier_phi,
    #     train_func=fit_optimized_word_cross_product,
    #     assess_reader=nli.NLIReader(snli['validation']))
    #
    # print(_['model'])
