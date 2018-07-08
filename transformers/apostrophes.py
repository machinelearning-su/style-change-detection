import re
from utils import print_progress_bar

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

def apostrophe_discrepancies(data, feature_names=[]):
    with open('data/external/apostrophe_words.txt', 'r') as f:
        apostrophes = list(map(lambda x: tuple(x.split(',')), f.read().splitlines()))

    vectors = []

    data_length = len(data)

    for i, entry in enumerate(data):
        entry = entry.lower()

        local = list(map(
            lambda x: float(min(entry.count(x[0]), entry.count(x[1]))),
            apostrophes
        ))

        vectors.append(local)

        print_progress_bar(i + 1, data_length, description = 'apostrophe_discrepancies')

    feature_names.extend([', '.join(a) for a in apostrophes])

    return vectors
