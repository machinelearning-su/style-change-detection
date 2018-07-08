from utils import print_progress_bar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

FUNCTION_WORDS_FILE = 'data/external/function_words/output_words.txt'

class NGrams():
    def __init__(self, train_x, pos_tag_x):
        with open(FUNCTION_WORDS_FILE) as f:
            self.function_words = f.read().splitlines()

        self.stop_word_vect = TfidfVectorizer(max_features=20)
        self.stop_word_vect.fit([self.only_stop_words(x) for x in train_x])

        self.function_word_vect = TfidfVectorizer(max_features=50)
        self.function_word_vect.fit([self.only_function_words(x) for x in train_x])

        self.word_vect = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
        self.word_vect.fit(train_x)

        self.pos_tag_vect = TfidfVectorizer(
            max_features=50, ngram_range=(2, 3))
        self.pos_tag_vect.fit(pos_tag_x)

    def get_stop_words(self, X, feature_names=[]):
        transformed = []
        vect = self.stop_word_vect

        for i, doc in enumerate(X):
            segments = [self.only_stop_words(s) for s in doc]
            transformed.append(vect.transform(segments).toarray())

            print_progress_bar(i + 1, len(X), description = 'stop words')

        feature_names.extend(vect.get_feature_names())

        return transformed

    def get_function_words(self, X, feature_names=[]):
        transformed = []
        vect = self.function_word_vect

        for i, doc in enumerate(X):
            segments = [self.only_function_words(s) for s in doc]
            transformed.append(vect.transform(segments).toarray())

            print_progress_bar(i + 1, len(X), description = 'function words')

        feature_names.extend(vect.get_feature_names())

        return transformed

    def get_pos_tags(self, X, feature_names=[]):
        transformed = []
        vect = self.pos_tag_vect

        for i, doc in enumerate(X):
            transformed.append(vect.transform(doc).toarray())

            print_progress_bar(i + 1, len(X), description = 'pos tags')

        feature_names.extend(vect.get_feature_names())

        return transformed

    def get_word_tfidf(self, X, feature_names=[]):
        transformed = []
        vect = self.word_vect

        for i, doc in enumerate(X):
            transformed.append(vect.transform(doc).toarray())

            print_progress_bar(i + 1, len(X), description = 'tfidf')

        feature_names.extend(vect.get_feature_names())

        return transformed

    def only_stop_words(self, text):
        words = word_tokenize(text.lower())
        stop_words = stopwords.words('english')

        return ' '.join(filter(lambda w: w in stop_words, words))

    def only_function_words(self, text):
        words = word_tokenize(text.lower())

        return ' '.join(filter(lambda w: w in self.function_words, words))

    @staticmethod
    def to_pos_tags(text):
        words = pos_tag(word_tokenize(text.lower()))

        return ' '.join([t for w, t in words])

def global_ngrams(X, vect, feature_names=[]):
    transformed = []

    for i, doc in enumerate(X):
        transformed.append(vect.transform(doc).toarray())
        print_progress_bar(i + 1, len(X), description = 'ngrams')

    feature_names.extend(vect.get_feature_names())

    return transformed
