import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

from models.base_estimator import BaseEstimator
from transformers import gmm
from chunkers import sliding_sent_chunks
from features import lexical


class MLP_GMM(BaseEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'MLP GMM',
            'mlp': {
                'fit': {
                    'batch_size': 32,
                    'epochs': 10,
                    'verbose': 1,
                    'validation_split': 0.1
                },
                'predict': {
                    'batch_size': 32,
                    'verbose': 1
                }
            },
            'gmm': {
                'pca': True,
                'n_components': 3,
                'covariance_type': 'full'
            }
        }

    def pipeline(self, X):
        return gmm(lexical(sliding_sent_chunks(X)), self.params['gmm'])

    def fit(self, train_x, train_y, train_positions):
        num_classes = np.max(train_y) + 1

        train_x = self.pipeline(train_x)
        train_y = keras.utils.to_categorical(train_y, num_classes=num_classes)

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(
            train_x.shape[1],), activation='sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        print('Fitting MLP model...')
        self.model.fit(train_x, train_y, **self.params['mlp']['fit'])

    def predict(self, test_x):
        test_x = self.pipeline(test_x)

        predictions = self.model.predict(
            test_x, **self.params['mlp']['predict'])

        return predictions.argmax(axis=-1)

    def get_grid_params(self):
        return {
            'gmm__pca': (True, False),
            'gmm__n_components': (2, 3, 4, 5)
        }
