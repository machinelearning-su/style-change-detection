from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


import numpy as np
import re
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

def tokenize(s): return re_tok.sub(r' \1 ', s).split()

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
COMMENT = "comment_text"
LIST_CLASSES = ["is_multi_author"]
MODELS_FP = 'models/'
TRAIN_FILENAME = '/pan_data/train.csv'
TEST_FILENAME = '/pan_data/test.csv'
LIST_LOGITS = ['logits_' + c for c in LIST_CLASSES]
TRAIN_SLIM_FILENAME = '/pan_data/train.csv'
SAMPLE_SUBMISSION_FILENAME = '/pan_data/sample_submission.csv'

import os

tokenizer = TweetTokenizer()

class Config:
    do_preprocess = True
    root = '/output/'
    fn_out_train = 'l2_train_data.csv'
    fp_out_train = root + fn_out_train
    fn_out_test = 'test_folded.csv'
    fp_out_test = root + fn_out_test
    fold_count = 10
    levels = ['word','char']

cfg = Config()

train = pd.read_csv(TRAIN_SLIM_FILENAME)
test = pd.read_csv(TEST_FILENAME)
subm = pd.read_csv(SAMPLE_SUBMISSION_FILENAME)


train['none'] = 1-train[LIST_CLASSES].max(axis=1)


train[COMMENT].fillna(NAN_WORD, inplace=True)
test[COMMENT].fillna(NAN_WORD, inplace=True)

def pr(y_i, y, feature):
    p = feature[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


fold_size = train.shape[0] // 10
X = train
Y = train[LIST_CLASSES].values

preds_test_list = []
preds_valid = np.zeros((len(train), len(LIST_CLASSES)))
res_y = np.zeros((len(train), len(LIST_CLASSES)))
for fold_id in range(0, 10):

    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    X_train = pd.concat([X[:fold_start], X[fold_end:]])
    X_valid = X[fold_start:fold_end]

    #reinitialize Vectorizer

    if 'word' in cfg.levels:
        word_vectorizer = TfidfVectorizer(ngram_range=(1, 2),
                                          tokenizer=tokenize,
                                          lowercase=True,
                                          min_df=3,
                                          max_df=0.9,
                                          strip_accents='unicode',
                                          use_idf=1,
                                          smooth_idf=1,
                                          sublinear_tf=1)

        print('fitting word Tfidf for fold %s' %fold_id)
        train_word_features = word_vectorizer.fit_transform(X_train[COMMENT])
        valid_word_features = word_vectorizer.transform(X_valid[COMMENT])
        test_word_features = word_vectorizer.transform(test[COMMENT])

    if 'char' in cfg.levels:
        char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                          lowercase=False,
                                          strip_accents='unicode',
                                          analyzer='char',
                                          ngram_range=(1, 4),
                                          max_features=30000)

        print('fitting char Tfidf for fold %s' % fold_id)
        train_char_features = char_vectorizer.fit_transform(X_train[COMMENT])
        valid_char_features = char_vectorizer.transform(X_valid[COMMENT])
        test_char_features = char_vectorizer.transform(test[COMMENT])

    for i, j in enumerate(LIST_CLASSES):
        print('fit', j)
        y = X_train[j].values
        if 'char' in cfg.levels:
            r1 = np.log(pr(1, y, train_char_features) / pr(0, y, train_char_features))
            x_nb1 = train_char_features.multiply(r1)
        if 'word' in cfg.levels:
            r2 = np.log(pr(1, y, train_word_features) / pr(0, y, train_word_features))
            x_nb2 = train_word_features.multiply(r2)

        if 'char' in cfg.levels and 'word' in cfg.levels:
            valid_features = hstack([valid_word_features.multiply(r2), valid_char_features.multiply(r1)])
            test_features = hstack([test_word_features.multiply(r2), test_char_features.multiply(r1)])
            x_nb = hstack([x_nb1, x_nb2])
        elif 'char' in cfg.levels:
            x_nb = x_nb1
            valid_features = valid_char_features.multiply(r1)
            test_features = test_char_features.multiply(r1)
        else:
            x_nb = x_nb2
            valid_features = valid_word_features.multiply(r2)
            test_features = test_word_features.multiply(r2)

        m = LogisticRegression(C=4, dual=True)
        m.fit(x_nb, y)

        preds_valid[fold_start:fold_end, i] = m.predict_proba(valid_features)[:, 1]
        res_y[fold_start:fold_end, i] = Y[fold_start:fold_end, i]
        preds_test_list.append(m.predict_proba(test_features)[:, 1])


l2_data = pd.DataFrame(columns=LIST_LOGITS+LIST_CLASSES)
l2_data[LIST_LOGITS] = pd.DataFrame(preds_valid)
l2_data[LIST_CLASSES] = pd.DataFrame(res_y)
l2_data.to_csv(cfg.fp_out_train)

preds_test_list2 = [np.array(preds_test_list[i:i+6]).T for i in range(10)]
test_predicts = np.ones(preds_test_list2[0].shape)
for fold_predict in preds_test_list2:
    test_predicts *= fold_predict

test_predicts **= (1. / len(preds_test_list2))

new_submission = pd.read_csv(SAMPLE_SUBMISSION_FILENAME)
new_submission[LIST_CLASSES] = test_predicts
new_submission.to_csv("/output/nbsvm_10fold.csv", index=False)