import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

from models.base_estimator import BaseEstimator

class MLP(BaseEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'Basic MLP',
            'tokenizer': {
                'num_words': 50000,
                'filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                'lower': True,
                'split': " ",
                'char_level': False,
                'oov_token': None
            },
            'mlp': {
                'fit':{
                    'batch_size': 32,
                    'epochs': 15,
                    'verbose': 1,
                    'validation_split': 0.1
                },
                'predict': {
                    'batch_size': 32,
                    'verbose': 0
                }
            }
        }

    def fit_with_test(self, train_x, train_y, train_positions, test_x):
        self.fit(train_x, train_y, train_positions, test_x)

    def fit(self, train_x, train_y, train_positions, test_x=None):
        num_classes = np.max(train_y) + 1

        self.tokenizer = keras.preprocessing.text.Tokenizer(**self.params['tokenizer'])
        all_text = train_x
        if test_x:
            all_text = train_x + test_x
        self.tokenizer.fit_on_texts(all_text)

        train_x = self.tokenizer.texts_to_matrix(train_x, mode='tfidf')
        train_y = keras.utils.to_categorical(train_y, num_classes = num_classes)

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(self.params['tokenizer']['num_words'],), activation='sigmoid'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print('Fitting MLP model...')
        self.model.fit(train_x, train_y, **self.params['mlp']['fit'])

    def predict(self, test_x):
        test_x = self.tokenizer.texts_to_matrix(test_x, mode='tfidf')

        predictions = self.model.predict(test_x, **self.params['mlp']['predict'])

        return predictions.argmax(axis=-1)
