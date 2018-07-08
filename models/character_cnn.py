import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Embedding, Input, Concatenate, Conv1D
from keras.layers import  GlobalMaxPool1D, Dropout, SpatialDropout1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from keras.layers import GlobalAveragePooling1D, Reshape, Dense, multiply
from models.base_estimator import BaseEstimator

IGNORE_ERRORS = 'ignore'
ASCII_AS_ENCODE_DECODE = 'ascii'
EMPTY_STRING = ""
CLASS = "is_multi_author"

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def flatten(list):
    return [item for sublist in list for item in sublist]

def preprocess(str):
    return str.encode(ASCII_AS_ENCODE_DECODE, errors=IGNORE_ERRORS).decode(ASCII_AS_ENCODE_DECODE, errors=IGNORE_ERRORS).lower()

class CharacterCNN(BaseEstimator):

    def __init__(self):
        self.params = {
            "MAX_LEN" : 1280,
            "BATCH_SIZE" : 256,
            "EPOCHS" : 200000000,
            "MAGIC_SEED" : 42,
            "BAGGING_SEED" : 1000000007,
            "NUMBER_OF_BAGGINGS" : 5
        }
        np.random.seed(self.params["MAGIC_SEED"])

    # We are using validation log loss for early stopping strategy, re-fitting strategy is better for unstable and small data
    def _train_model(self, model, batch_size, train_x, train_y, val_x, val_y):
        best_loss = -1
        best_weights = None
        best_epoch = 0

        current_epoch = 0

        while True:
            model.fit(train_x, train_y, batch_size=batch_size, epochs=1, verbose=2) # set verbose to 1 for fancy progress bar
            y_pred = model.predict(val_x, batch_size=batch_size)

            total_loss = 0
            for j in range(1):
                loss = log_loss(val_y, y_pred[:, j])
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

    def enhance_important_filters(self, input, ratio=16):
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


    def get_character_cnn(self, max_features):
        inp = Input(shape=(self.params["MAX_LEN"],), name="text")

        x = Embedding(max_features, 300)(inp)
        x = SpatialDropout1D(0.1)(x)

        c1 = Conv1D(64, 11, activation="relu")(x)

        c1 = self.enhance_important_filters(c1)

        c_1 = GlobalMaxPool1D()(c1)

        x = Dropout(0.1)(Dense(128, activation="relu")(c_1))
        x = Dropout(0.1)(Dense(128, activation="relu")(x))

        x = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=[inp], outputs=x)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def predict(self, test_x, return_probability=False):
        list_sentences_test = [preprocess(text) for text in test_x]

        list_tokenized_test = self.tokenizer.texts_to_sequences(list_sentences_test)
        X_test = {}
        X_test['text'] = sequence.pad_sequences(list_tokenized_test, maxlen=self.params["MAX_LEN"],
                                                padding='post', truncating='post')
        max_features_test = np.unique(flatten(X_test['text'])).shape[0] + 1
        print('max_features_test:', max_features_test)

        predict = np.zeros((len(test_x), 1))
        for model in self.models:
            predict_on_test = model.predict(X_test,
                                            batch_size=self.params["BATCH_SIZE"]) * (1.0 / self.params["NUMBER_OF_BAGGINGS"])
            predict += predict_on_test  # 5 times, used in our final prediction

        sample_submission = pd.DataFrame()
        sample_submission[CLASS] = predict[:, 0]

        if not return_probability:
            sample_submission[sample_submission[CLASS] < 0.5] = 0
            sample_submission[sample_submission[CLASS] >= 0.5] = 1

        return sample_submission[CLASS]

    def predict_proba(self, test_x):
        probabilities = self.predict(test_x, return_probability=True)
        return np.array(map(lambda x: [x], probabilities))

    def fit(self, train_x, train_y, train_positions):
        return self.fit_with_test(train_x, train_y, train_positions, None)

    def fit_with_test(self, train_x, train_y, train_positions, test_x):
        list_sentences_train = [preprocess(text) for text in train_x]
        self.tokenizer = Tokenizer()

        if test_x:
            list_sentences_test = [preprocess(text) for text in test_x]
            self.tokenizer.fit_on_texts(list(list_sentences_train) + list(list_sentences_test))
        else:
            self.tokenizer.fit_on_texts(list(list_sentences_train))

        list_tokenized_train = self.tokenizer.texts_to_sequences(list_sentences_train)
        list_tokenized_test = self.tokenizer.texts_to_sequences(list_sentences_test)

        X_train = {}
        X_test = {}

        X_train['text'] = sequence.pad_sequences(list_tokenized_train, maxlen=self.params["MAX_LEN"],
                                                 padding='post', truncating='post')
        X_test['text'] = sequence.pad_sequences(list_tokenized_test, maxlen=self.params["MAX_LEN"], padding='post', truncating='post')

        max_features = np.unique(flatten(X_train['text'])).shape[0] + 1
        print('max_features_train:', max_features)
        max_features_test = np.unique(flatten(X_test['text'])).shape[0] + 1
        print('max_features_test:', max_features_test)

        max_features = max_features + max_features_test
        print('STARTING...')

        self.kf = StratifiedKFold(n_splits=self.params["NUMBER_OF_BAGGINGS"],
                                  shuffle=True, random_state=self.params["BAGGING_SEED"])
        totalFoldAccuracy = 0.0
        self.models = []
        train_y = np.array(train_y)
        for (f, (train_index, valid_index)) in enumerate(self.kf.split(X_train['text'], train_y)):
            train_part = X_train['text'][train_index]
            valid_part = X_train['text'][valid_index]

            y_train = train_y[train_index]
            y_valid = train_y[valid_index]

            model = self.get_character_cnn(max_features)
            model = self._train_model(model, self.params["BATCH_SIZE"],
                                      train_part, y_train, valid_part, y_valid)

            self.models.append(model)

            predict_on_validation = model.predict(valid_part,
                                                  batch_size=self.params["BATCH_SIZE"])

            temp = pd.DataFrame()
            temp[CLASS] = predict_on_validation[:,0]
            temp[temp[CLASS] < 0.5] = 0
            temp[temp[CLASS] >= 0.5] = 1

            verify = pd.DataFrame()
            verify[CLASS] = y_valid

            foldAccuracy = (100 * (temp[CLASS] == verify[CLASS]).sum()) / len(y_valid)
            totalFoldAccuracy += foldAccuracy
            print("Fold accuracy = {0}".format(foldAccuracy))

        print("Total Bagging Accuracy for 5 folds is {0}".format(totalFoldAccuracy / self.params["NUMBER_OF_BAGGINGS"]))
