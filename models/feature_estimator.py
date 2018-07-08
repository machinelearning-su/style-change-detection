import numpy as np

from models.base_estimator import BaseEstimator
from transformers import gmm, max_diff, stdev
from transformers import num_paragraphs
from chunkers import sliding_sent_chunks, sent_chunks, word_chunks, char_chunks
from features import lexical, global_ngrams, readability, processed_tags

from sklearn.preprocessing import minmax_scale

from sklearn.feature_extraction.text import TfidfVectorizer
from features.word_frequency import WordFrequency
from features.ngrams import NGrams
from preprocessing.basic_preprocessor import BasicPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel


wf = WordFrequency()
preprocessor = BasicPreprocessor()


class FeatureEstimator(BaseEstimator):
    def pipeline(self, X, pos_tag_x, fit_scalers):
        feature_names = []

        X = [preprocessor.process_text(x) for x in X]

        X_word_chunks = word_chunks(X, chunks=4, process=True, sliding=True)
        X_pos_chunks = word_chunks(pos_tag_x, chunks=4, process=True, sliding=True)

        lexical_features = max_diff(lexical(X_word_chunks, feature_names))

        stop_word_features = max_diff(self.ngrams.get_stop_words(
            X_word_chunks,
            feature_names
        ))

        function_word_features = max_diff(self.ngrams.get_function_words(
            X_word_chunks,
            feature_names
        ))

        pos_tag_features = max_diff(self.ngrams.get_pos_tags(
            X_pos_chunks,
            feature_names
        ))

        process_tag_features = max_diff(processed_tags(
            X_word_chunks,
            feature_names
        ))

        word_frequency = max_diff(wf.average_word_frequency(X_word_chunks, feature_names))
        readability_features = max_diff(readability(X_word_chunks, feature_names=feature_names))
        #tfidf = max_diff(self.ngrams.get_word_tfidf(X_word_chunks, feature_names))
        num_par = num_paragraphs(X, feature_names)

        X = np.concatenate((lexical_features, stop_word_features,
                            function_word_features, pos_tag_features,
                            process_tag_features, word_frequency,
                            readability_features, num_par), axis=1)

        X = minmax_scale(X)

        return X, feature_names

    def fit(self, train_x, train_y, train_positions):
        pos_tag_x = [NGrams.to_pos_tags(x) for x in train_x]
        self.ngrams = NGrams(train_x, pos_tag_x)

        self.train_x, names = self.pipeline(
            train_x, pos_tag_x, fit_scalers=True)

        self.print_feature_importance(self.train_x, train_y, names)

    def predict(self, test_x):
        pos_tag_x = [NGrams.to_pos_tags(x) for x in test_x]
        test_x, _ = self.pipeline(test_x, pos_tag_x, fit_scalers=False)

        return self.model.predict(test_x).tolist()

    
    def print_feature_importance(self, train_x, train_y, names):
        params =  {
                'inverse_strength' : 2.0,
                'solver' : 'sag',
                'minimum_feature_weight' : 0.3,
        }
        logreg_model = LogisticRegression(C=params['inverse_strength'],
                                       solver=params['solver'])
        selector = SelectFromModel(logreg_model, threshold=params['minimum_feature_weight'])
        selector.fit(train_x, train_y)

        coef = selector.estimator_.coef_.reshape(-1)
        sorted_features = sorted(zip(map(lambda x: round(x, 4), coef), 
                names), reverse=True)
        best = [f for f in sorted_features if f[0]>=params['minimum_feature_weight']]
        print('-------------------------------------------')
        for f in best:
            print(f)
        print('-------------------------------------------')
