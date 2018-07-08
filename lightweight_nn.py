#import tensorflow as tf

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input, Concatenate, Conv1D, Activation, RepeatVector, Permute, multiply
from keras.layers import  GlobalMaxPool1D, Dropout, GlobalAveragePooling1D, MaxPooling1D, \
    SpatialDropout1D
from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
import re, gc
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from keras import backend as K
from sklearn.model_selection import StratifiedKFold

from keras.layers import GlobalAveragePooling1D, Reshape, Dense, multiply, add, TimeDistributed, Permute, Lambda, RepeatVector

MAX_LEN = 1560
BATCH_SIZE = 256
EPOCHS = 200000000
MAGIC_SEED = 42
BAGGING_SEED = 1000000007
NUMBER_OF_BAGGINGS = 5
IGNORE_ERRORS = 'ignore'
ASCII_AS_ENCODE_DECODE = 'ascii'
EMPTY_STRING = ""
LIST_CLASSES = ["is_multi_author"]

np.random.seed(MAGIC_SEED)

train = pd.read_csv("my_train.csv")
test = pd.read_csv("my_test.csv")

def preprocess(s):
    return s.encode(ASCII_AS_ENCODE_DECODE, errors=IGNORE_ERRORS).decode(ASCII_AS_ENCODE_DECODE, errors=IGNORE_ERRORS).lower()

def flatten(l): return [item for sublist in l for item in sublist]

# We are using validation log loss for early stopping strategy, re-fitting strategy is better for unstable and small data
def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1, verbose=2) # set verbose to 1 for fancy progress bar
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        for j in range(1):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= 1.

        if (np.isnan(total_loss)):
            break

        current_epoch += 1

        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 5:
                break

    model.set_weights(best_weights)
    return model

def enhance_important_filters(input, ratio=16):
    init = input
    channel_axis = -1
    filters = init._keras_shape[channel_axis]
    shape = (1, filters)

    se = GlobalAveragePooling1D()(init)
    se = Reshape(shape)(se)
    se = Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(se)

    output = multiply([init, se])

    return output


def get_character_cnn():
    inp = Input(shape=(MAX_LEN,), name="text")

    x = Embedding(max_features, 300)(inp)
    x = SpatialDropout1D(0.1)(x)


    c1 = Conv1D(64, 11, activation="relu")(x)

    c1 = enhance_important_filters(c1)

    c_1 = GlobalMaxPool1D()(c1)

    x = Dropout(0.1)(Dense(128, activation="relu")(c_1))
    x = Dropout(0.1)(Dense(128, activation="relu")(x))

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[inp], outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

train['text'] = train.text.fillna(EMPTY_STRING).apply(preprocess)
test['text'] = test.text.fillna(EMPTY_STRING).apply(preprocess)

list_sentences_train = train["text"].fillna(EMPTY_STRING).values
y = train[LIST_CLASSES].values
list_sentences_test = test["text"].fillna(EMPTY_STRING).values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_train = {}
X_test = {}

X_train['text'] = sequence.pad_sequences(list_tokenized_train, maxlen=MAX_LEN, padding='post', truncating='post')
X_test['text'] = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_LEN, padding='post', truncating='post')

max_features = np.unique(flatten(X_train['text'])).shape[0] + 1
print('max_features_train:', max_features)
max_features_test = np.unique(flatten(X_test['text'])).shape[0] + 1
print('max_features_test:', max_features_test)

max_features = max_features + max_features_test
print('STARTING...')

scores = []
predict = np.zeros((test.shape[0], 1))

kf = StratifiedKFold(n_splits=NUMBER_OF_BAGGINGS, shuffle=True, random_state=BAGGING_SEED)

totalFoldAccuracy = 0.0
for (f, (train_index, valid_index)) in enumerate(kf.split(X_train['text'], train['is_multi_author'])):

    train_part = X_train['text'][train_index]
    valid_part = X_train['text'][valid_index]

    y_train = y[train_index]
    y_valid = y[valid_index]

    model = get_character_cnn()
    model = _train_model(model, BATCH_SIZE, train_part, y_train, valid_part, y_valid)

    predict_on_test = model.predict(X_test, batch_size=BATCH_SIZE) * (1.0 / NUMBER_OF_BAGGINGS)  # Bagging = 5 => * 1/5
    predict_on_validation = model.predict(valid_part, batch_size=BATCH_SIZE)

    predict += predict_on_test  # 5 times, used in our final prediction

    temp = pd.DataFrame()
    temp['is_multi_author'] = np.zeros(len(y_valid))
    temp[LIST_CLASSES] = predict_on_validation
    temp[temp.is_multi_author < 0.5] = 0
    temp[temp.is_multi_author >= 0.5] = 1

    verify = pd.DataFrame()
    verify['is_multi_author'] = np.zeros(len(y_valid))
    verify['is_multi_author'] = y_valid

    foldAccuracy = (100 * (temp['is_multi_author'] == verify['is_multi_author']).sum()) / len(y_valid)
    totalFoldAccuracy += foldAccuracy
    print("Fold accuracy = {0}".format(foldAccuracy))


print("Total Bagging Accuracy for 5 folds is {0}".format(totalFoldAccuracy/5))

sample_submission = pd.DataFrame()

for c in LIST_CLASSES:
    sample_submission[c] = np.zeros(len(test))

sample_submission[LIST_CLASSES] = predict

#sample_submission[ sample_submission.is_multi_author < 0.5] = 0
#sample_submission[ sample_submission.is_multi_author >= 0.5] = 1

# outputing probabilities :)
#sample_submission.to_csv("/output/char_cnn_pred.csv", index=False)
