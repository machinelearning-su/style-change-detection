from models.feature_estimator import FeatureEstimator
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras import regularizers
from keras.models import Model
from keras.optimizers import Adam

import numpy as np
from models.base_estimator import BaseEstimator
from chunkers import word_chunks, char_chunks
from features import lexical, global_ngrams, readability, processed_tags
from sklearn.feature_extraction.text import TfidfVectorizer
from features.word_frequency import WordFrequency
from features.ngrams import NGrams
from preprocessing.basic_preprocessor import BasicPreprocessor

wf = WordFrequency()
preprocessor = BasicPreprocessor()

TRAIN_X_FILE = 'data/feature_vectors/train_x.npy'
TRAIN_Y_FILE = 'data/feature_vectors/train_y.npy'

class LSTM_model(BaseEstimator):
    def __init__(self):
        self.params = {
            'model_description': 'LSTM',
            'lstm_params': {
                'epochs': 5,
                'batch_size': 64
            },
            'train_from_file': True
        }

    def get_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(256, dropout=0.2, return_sequences = True, input_shape=input_shape))
        model.add(LSTM(256, dropout=0.2, return_sequences = False))
        model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))
        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model

    def pipeline(self, X, pos_tag_x):
        X = [preprocessor.process_text(x) for x in X]

        X_word_chunks = word_chunks(X, n=300, process=True, sliding=True)
        X_char_chunks = char_chunks(X, n=2000, sliding=True)
        X_pos_chunks = word_chunks(pos_tag_x, n=300, process=True, sliding=True)

        max_segments = 20
        lexical_features = sequence.pad_sequences(lexical(X_word_chunks), maxlen=max_segments)
        stop_word_features = sequence.pad_sequences(self.ngrams.get_stop_words(X_word_chunks), maxlen=max_segments)
        function_word_features = sequence.pad_sequences(self.ngrams.get_function_words(X_word_chunks), maxlen=max_segments)
        pos_tag_features = sequence.pad_sequences(self.ngrams.get_pos_tags(X_pos_chunks), maxlen=max_segments)
        word_frequency = sequence.pad_sequences(wf.average_word_frequency(X_word_chunks), maxlen=max_segments)
        readability_features = sequence.pad_sequences(readability(X_word_chunks), maxlen=max_segments)

        return np.concatenate([lexical_features, stop_word_features,
                            function_word_features, pos_tag_features,
                            word_frequency,
                            readability_features], axis=2)

    def fit(self, train_x, train_y, train_positions):
        pos_tag_x = [NGrams.to_pos_tags(x) for x in train_x]
        self.ngrams = NGrams(train_x, pos_tag_x)

        if self.params['train_from_file']:
            print("Loading features from file...")
            X = np.load(TRAIN_X_FILE)
            train_y = np.load(TRAIN_Y_FILE)
        else:
            X = self.pipeline(train_x, pos_tag_x)

        self.model = self.get_model((X.shape[1], X.shape[2]))
        print('Fitting LSTM model...')

        self.model.fit(X, np.array(train_y), **self.params['lstm_params'])

    def predict(self, test_x):
        pos_tag_x = [NGrams.to_pos_tags(x) for x in test_x]
        test_x = self.pipeline(test_x, pos_tag_x)

        predictions = self.model.predict_classes(test_x)
        return predictions.flatten()
