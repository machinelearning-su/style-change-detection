import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import time
from scipy.sparse import hstack
from scipy.special import logit, expit
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def tokenize(s): return re_tok.sub(r' \1 ', s).split()


def pr(y_i, y, x):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


def get_mdl(y,x, c0 = 4):
    y = y.values
    r = np.log(pr(1,y,x) / pr(0,y,x))
    m = LogisticRegression(C= c0, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


model_type = 'lrchar'
todate = time.strftime("%d%m")

# read data
train = pd.read_csv('/pan_data/train.csv')
test = pd.read_csv('/pan_data/test.csv')
subm = pd.read_csv('/pan_data/sample_submission.csv')

id_train = train['id'].copy()
id_test = test['id'].copy()

# add empty label for None
label_cols = ['is_multi_author']
train['none'] = 1-train[label_cols].max(axis=1)
# fill missing values
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

n = train.shape[0]

word_vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    min_df = 5,
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3))

all1 = pd.concat([train[COMMENT], test[COMMENT]])
word_vectorizer.fit(all1)
xtrain1 = word_vectorizer.transform(train[COMMENT])
xtest1 = word_vectorizer.transform(test[COMMENT])

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    min_df = 3,
    ngram_range=(1, 6))

all1 = pd.concat([train[COMMENT], test[COMMENT]])
char_vectorizer.fit(all1)

xtrain2 = char_vectorizer.transform(train[COMMENT])
xtest2 = char_vectorizer.transform(test[COMMENT])

nfolds = 6
xseed = 777 # lucky shot :D
cval = 4

xtrain = hstack([xtrain1, xtrain2], format='csr')
xtest = hstack([xtest1,xtest2], format='csr')
ytrain = np.array(train[label_cols].copy())

del xtrain1, xtrain2, xtest1, xtest2
gc.collect()

skf = StratifiedKFold(n_splits= nfolds, random_state= xseed)

predval = np.zeros((xtrain.shape[0], len(label_cols)))
predfull = np.zeros((xtest.shape[0], len(label_cols)))

for (lab_ind,lab) in enumerate(label_cols):
    y = train[lab].copy()
    print('label:' + str(lab_ind))
    for (f, (train_index, test_index)) in enumerate(skf.split(xtrain, y)):
        # split
        x0, x1 = xtrain[train_index], xtrain[test_index]
        y0, y1 = y[train_index], y[test_index]
        m,r = get_mdl(y0,x0, c0 = cval)
        predval[test_index,lab_ind] = m.predict_proba(x1.multiply(r))[:,1]
        m,r = get_mdl(y,xtrain, c0 = cval)
        predfull[:,lab_ind] += m.predict_proba(xtest.multiply(r))[:,1]
        print('fit:'+ str(lab) + ' fold:' + str(f))

    gc.collect()
predfull /= nfolds

# store prval
prval = pd.DataFrame(predval)
prval.columns = label_cols
prval['id'] = id_train
prval.to_csv('/output/prval_'+model_type+'x'+str(cval)+'f'+str(nfolds)+'_'+todate+'.csv', index= False)

# store prfull
prfull = pd.DataFrame(predfull)
prfull.columns = label_cols
prfull['id'] = id_test
prfull.to_csv('/output/prfull_'+model_type+'x'+str(cval)+'f'+str(nfolds)+'_'+todate+'.csv', index= False)

# store submission
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(prfull, columns = label_cols)], axis=1)
submission.to_csv('/output/sub_'+model_type+'x'+str(cval)+'f'+str(nfolds)+'_'+todate+'.csv', index= False)