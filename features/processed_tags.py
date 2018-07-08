from nltk import pos_tag
from nltk.tokenize import word_tokenize
import numpy as np
from utils import print_progress_bar
from preprocessing.basic_preprocessor import BasicPreprocessor

preprocessor = BasicPreprocessor()

def processed_tags(X, feature_names=[]):
    transformed = []

    for i, doc in enumerate(X):
        segments = []

        for entry in doc:
            words = word_tokenize(entry)
            word_count = len(words)
            word_analysis = dict.fromkeys(preprocessor.tags, 0)

            for word in words:
                for tag in preprocessor.tags:
                    if word == tag:
                        word_analysis[tag] += 1

            segments.append([word_analysis[key]/word_count for key in preprocessor.tags])

        transformed.append(segments)

        print_progress_bar(i + 1, len(X), description = 'processed tags')

    feature_names.extend(preprocessor.tags)

    return np.array(transformed)
