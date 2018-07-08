from nltk.tokenize import word_tokenize
import re
import math
from sklearn.preprocessing import MinMaxScaler
from utils import print_progress_bar

COMMON_WORDS_FILE = 'data/external/common_words/google-books-common-words.txt'

class WordFrequency():
    def __init__(self):
        self.word_class = {}
        with open(COMMON_WORDS_FILE) as f:
            for line in f:
                key, val = line.split()
                self.word_class[key.lower()] = math.log2(53097401461/float(val))

    def average_word_frequency(self, X, feature_names=[]):
        transformed = []

        for i, doc in enumerate(X):
            segments = []

            for entry in doc:
                class_sum = 0
                word_count = 0
                uncommon = 0
                entry = entry.lower()
                for w in word_tokenize(entry):
                    w = re.sub('[^a-zA-Z]+', '', w)
                    if not w: continue

                    word_count+=1
                    word_class = self.word_class.get(w, 20)
                    if word_class == 20:
                        uncommon += 1
                    class_sum += word_class

                segments.append([class_sum/word_count, uncommon/word_count])

            transformed.append(segments)
            print_progress_bar(i + 1, len(X), description = 'word frequency')

        feature_names.extend(['average_word_class', 'uncommon_words'])

        return transformed
